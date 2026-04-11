"""
Reranking RAG System — FAISS Retrieval + Cross-Encoder Reranking

The Problem with Naive RAG:
  FAISS uses "bi-encoders" — it embeds the question and each document chunk
  *separately*, then compares their vectors. This is fast, but it misses
  fine-grained relationships between the question and the document because
  they were never looked at together.

  Think of it like a librarian who checks each book's title tag independently
  against a list of keywords, rather than actually reading the question and
  the book together.

The Fix — Two-Stage Retrieval:
  Stage 1 (Fast but rough): FAISS fetches a large pool of candidates (e.g. 10)
  Stage 2 (Slow but accurate): Cross-Encoder rescores each candidate

What is a Cross-Encoder?
  It reads the question AND a candidate chunk TOGETHER in a single pass,
  letting it understand how they relate to each other in full context.
  This produces much more accurate relevance scores than bi-encoders.

  The trade-off: it's too slow to run on every chunk in the knowledge base,
  so we only use it as a second-pass filter on the top candidates from FAISS.

  It's like having the librarian quickly scan the shelves first (FAISS),
  then carefully read the most promising books (cross-encoder) before
  recommending the best 3.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2 (~67MB, runs on CPU)

Pipeline:
  Question -> FAISS fetches top-10 candidates
           -> Cross-encoder scores each (question, candidate) pair
           -> Keep only the top-3 most relevant chunks
           -> LLM generates answer from those 3 chunks
"""

import asyncio
import logging
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

from eval_framework.config import get_settings
from eval_framework.systems.shared import SharedIndex
from eval_framework.types import QAPair, SystemOutput

logger = logging.getLogger(__name__)

# Rough cost per output token for Groq-hosted Llama models
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000

# Cross-encoder model: small, fast, good quality. ~67MB download, runs on CPU.
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRAGSystem:
    """
    Retrieve a large candidate pool with FAISS, then narrow it down
    with a cross-encoder that reads question + chunk together.

    Expected improvement over Naive RAG: better precision — the top-k
    chunks that reach the LLM are more tightly relevant to the question.
    """

    def __init__(
        self,
        index: SharedIndex,
        top_k: int = 3,
        candidate_k: int = 10,
        model_name: str = "llama-3.3-70b-versatile",
        reranker: CrossEncoder | None = None,
    ):
        """
        Args:
            index:       The shared FAISS + BM25 index.
            top_k:       Final number of chunks sent to the LLM after reranking.
            candidate_k: How many chunks FAISS fetches as the candidate pool.
                         Should be larger than top_k (e.g. top_k=3, candidate_k=10).
            model_name:  Groq model for answer generation.
            reranker:    Optional pre-loaded CrossEncoder. Pass this when running
                         multiple systems together so the model loads only once.
        """
        self._index = index
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.model_name = model_name

        settings = get_settings()

        # LLM for generating the final answer
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Load the cross-encoder for reranking.
        # If one was passed in (pre-loaded externally), reuse it to save time.
        if reranker is not None:
            self._reranker = reranker
        else:
            print(f"Loading cross-encoder reranker ({_RERANKER_MODEL})...")
            self._reranker = CrossEncoder(_RERANKER_MODEL)
            print("Reranker ready")

        # Same grounding prompt as other RAG systems
        self._prompt = ChatPromptTemplate.from_template(
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY the information in the context below. "
            "If the context does not contain enough information, say: "
            "'The document does not contain enough information to answer this question.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def _rerank(self, question: str, candidates: list) -> list:
        """
        Score each candidate chunk against the question using the cross-encoder,
        then return only the top-k most relevant chunks.

        Args:
            question:   The user's original question.
            candidates: List of Document objects from the FAISS candidate pool.

        Returns:
            The top-k Document objects, sorted by cross-encoder relevance score.
        """
        # Build (question, chunk_text) pairs for the cross-encoder.
        # The cross-encoder needs to see BOTH the question and the chunk together.
        pairs = [(question, doc.page_content) for doc in candidates]

        # Get relevance scores — higher score = more relevant
        scores = self._reranker.predict(pairs)

        # Zip scores with their documents, sort by score (highest first)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        # Return only the top-k documents (discard the scores)
        return [doc for _, doc in ranked[:self.top_k]]

    async def query(self, qa_pair: QAPair) -> SystemOutput:
        """
        Two-stage retrieval: FAISS candidate pool -> cross-encoder reranking -> LLM answer.

        Args:
            qa_pair: Contains the question to answer.

        Returns:
            SystemOutput with the answer, context, timing, and cost.
        """
        start = time.time()

        # --- Stage 1: FAISS fetches a large candidate pool ---
        # We ask for MORE chunks than we'll actually use (candidate_k > top_k).
        # This gives the cross-encoder a good pool to pick the best from.
        # Run in thread executor because FAISS is synchronous.
        candidates = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._index.vectorstore.similarity_search(
                qa_pair.question, k=self.candidate_k
            ),
        )

        # --- Stage 2: Cross-encoder reranks the candidates ---
        # This is CPU-bound (model inference), so also runs in a thread executor.
        source_docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._rerank(qa_pair.question, candidates)
        )

        # Join the top-k reranked chunks into a context block
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        # Generate the answer using only the best-ranked context
        messages = await self._prompt.ainvoke({"context": context, "question": qa_pair.question})
        response = await self._llm.ainvoke(messages)
        answer = response.content

        latency_ms = (time.time() - start) * 1000

        # Store context for the evaluators
        if context:
            qa_pair.context = context

        # Rough cost estimate: word count * 1.3 approximates token count
        output_tokens = len(answer.split()) * 1.3
        estimated_cost = output_tokens * _COST_PER_OUTPUT_TOKEN

        logger.info(
            f"RerankingRAG answered in {latency_ms:.0f}ms | "
            f"{self.candidate_k} candidates -> {len(source_docs)} after reranking"
        )

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "system": "reranking_rag",
                "candidates_fetched": self.candidate_k,     # pool size before reranking
                "chunks_after_rerank": len(source_docs),    # what the LLM actually sees
            },
        )
