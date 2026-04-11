"""
Hybrid RAG System — BM25 Keyword Search + FAISS Semantic Search

The Problem with Naive RAG (FAISS only):
  FAISS finds chunks that are semantically similar to the question — meaning
  chunks with similar *meaning*. But if the document uses different words
  than the question, the similarity score drops, even if the content is
  directly relevant.

  Example: Question asks "What is FAISS?" but the document says
  "Facebook AI Similarity Search (FAISS) is..."
  The word overlap is low, so semantic search might miss it.

The Fix — Two Search Methods:
  1. FAISS (semantic): Finds chunks with similar *meaning* using vector embeddings.
     Great for paraphrases and conceptual similarity.

  2. BM25 (keyword): Finds chunks containing the exact *words* from the question.
     Great for technical terms, proper nouns, and acronyms.

  Using both together gives better recall than either one alone.

How Results Are Merged — Reciprocal Rank Fusion (RRF):
  Each retriever returns a ranked list of chunks. Instead of just taking
  the top results from one list, RRF combines both ranked lists into a
  single merged ranking.

  Formula: score(chunk) = 1/(60 + rank_in_BM25) + 1/(60 + rank_in_FAISS)
  A chunk that ranks highly in BOTH lists gets a very high combined score.
  The constant 60 prevents the very top results from dominating too much.

Pipeline:
  Question -> BM25 search (top-8 candidates)
           -> FAISS search (top-8 candidates)
           -> RRF merge both ranked lists
           -> Keep top-3 for LLM
           -> Generate answer
"""

import asyncio
import logging
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from eval_framework.config import get_settings
from eval_framework.systems.shared import SharedIndex, reciprocal_rank_fusion
from eval_framework.types import QAPair, SystemOutput

logger = logging.getLogger(__name__)

# Rough cost per output token for Groq-hosted Llama models
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000


class HybridRAGSystem:
    """
    Retrieves candidates with both BM25 (keyword) and FAISS (semantic) search,
    then merges the two ranked lists using Reciprocal Rank Fusion.

    Expected improvement over Naive RAG: better recall on technical terms,
    exact names, and acronyms that pure semantic search might miss.
    """

    def __init__(
        self,
        index: SharedIndex,
        top_k: int = 3,
        candidate_k: int = 8,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        """
        Args:
            index:       The shared FAISS + BM25 index (built once, reused here).
            top_k:       Final number of chunks passed to the LLM after merging.
            candidate_k: How many chunks each search method retrieves before merging.
                         More candidates = better recall but slightly more processing.
            model_name:  Groq model for answer generation.
        """
        self._index = index
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.model_name = model_name

        settings = get_settings()

        # LLM for answer generation — low temperature for factual consistency
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Same grounding prompt as Naive RAG — ONLY the retrieval step differs
        self._prompt = ChatPromptTemplate.from_template(
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY the information in the context below. "
            "If the context does not contain enough information, say: "
            "'The document does not contain enough information to answer this question.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def _hybrid_retrieve(self, question: str) -> list:
        """
        Run BM25 + FAISS searches and merge their results with RRF.

        Returns:
            List of top-k Document objects ranked by combined relevance score.
        """
        chunks = self._index.chunks

        # --- BM25 Search (keyword-based) ---
        # Returns (Document, score, chunk_index) tuples — we only need the index
        bm25_results = self._index.bm25.retrieve(question, k=self.candidate_k)
        bm25_ranking = [idx for _, _, idx in bm25_results]
        # bm25_ranking = [3, 12, 7, ...] — chunk indices in order of BM25 relevance

        # --- FAISS Search (semantic/embedding-based) ---
        # Returns Document objects (not indices), so we need to map them back
        faiss_docs = self._index.vectorstore.similarity_search(
            question, k=self.candidate_k
        )
        # Build a lookup: chunk text -> its index in the full chunks list
        content_to_idx = {chunk.page_content: i for i, chunk in enumerate(chunks)}
        faiss_ranking = [content_to_idx.get(doc.page_content, -1) for doc in faiss_docs]
        # Remove any -1s (docs not found in lookup — shouldn't happen, but safe guard)
        faiss_ranking = [i for i in faiss_ranking if i >= 0]
        # faiss_ranking = [7, 3, 21, ...] — chunk indices in order of FAISS relevance

        # --- Merge with RRF ---
        # RRF takes both ranked lists and combines them into one final ranking.
        # Chunks appearing high in BOTH lists float to the top.
        merged_indices = reciprocal_rank_fusion([bm25_ranking, faiss_ranking])

        # Return the top_k chunks as Document objects
        return [chunks[i] for i in merged_indices[:self.top_k]]

    async def query(self, qa_pair: QAPair) -> SystemOutput:
        """
        Retrieve with BM25 + FAISS, merge results, then generate a grounded answer.

        Args:
            qa_pair: Contains the question to answer.

        Returns:
            SystemOutput with the answer, context, timing, and cost.
        """
        start = time.time()

        # _hybrid_retrieve is CPU-bound (BM25 and FAISS are synchronous).
        # Run it in a thread so other async coroutines aren't blocked.
        source_docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._hybrid_retrieve(qa_pair.question)
        )

        # Join the top-k chunks into a context block for the LLM
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        # Generate the answer using only the retrieved context
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

        logger.info(f"HybridRAG answered in {latency_ms:.0f}ms | {len(source_docs)} chunks (BM25+FAISS)")

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "system": "hybrid_rag",
                "chunks_retrieved": len(source_docs),
                "candidate_k": self.candidate_k,  # how many each retriever fetched before merging
            },
        )
