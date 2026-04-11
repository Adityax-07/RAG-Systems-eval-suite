"""
Advanced RAG System — The Best of All Worlds

This system combines THREE techniques to give the highest quality answers:

  Step 1 — Query Rewriting:
    Instead of searching with just the original question, we ask the LLM to
    rephrase it N different ways. This helps catch relevant chunks that use
    different vocabulary than the user's exact wording.
    Example: "What is RAG?" might also be searched as
             "How does Retrieval-Augmented Generation work?" etc.

  Step 2 — Hybrid Retrieval (BM25 + FAISS):
    For EACH query variant, we run two searches in parallel:
      - BM25: keyword-based (good at exact word matches)
      - FAISS: vector/semantic (good at meaning matches)
    Then we merge all these ranked lists using Reciprocal Rank Fusion (RRF),
    which gives us the best of both search styles across all query variants.

  Step 3 — Cross-Encoder Reranking:
    After hybrid retrieval we have many candidate chunks. The cross-encoder
    reads the question + each chunk together (not separately) and assigns
    a relevance score. This is slower but far more accurate than the initial
    retrieval, so it filters out noise before the LLM sees any context.

  Step 4 — Answer Generation:
    The top-k reranked chunks are joined into a context block and passed to
    the LLM to generate a grounded answer.

Why is this the "advanced" system?
  - More relevant context = more faithful, complete answers
  - Reranking filters out irrelevant chunks = less hallucination
  - Trade-off: ~3x more tokens and ~2x more latency vs naive RAG
"""

import asyncio
import logging
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

from eval_framework.config import get_settings
from eval_framework.systems.shared import SharedIndex, reciprocal_rank_fusion
from eval_framework.systems.query_rewriting import _parse_query_variants
from eval_framework.types import QAPair, SystemOutput

logger = logging.getLogger(__name__)

# Cost per output token for Llama models on Groq (used to estimate run cost)
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000

# The cross-encoder model we use for reranking — small but very accurate
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class AdvancedRAGSystem:
    """
    The most powerful RAG system in this benchmark.

    Pipeline summary:
        User Question
            -> Query Rewriting (1 question becomes N)
            -> Hybrid Retrieval (BM25 + FAISS for every variant)
            -> RRF Merge (combine all ranked lists into one)
            -> Cross-Encoder Reranking (score each chunk more carefully)
            -> LLM Answer Generation (using only the top-k chunks)
    """

    def __init__(
        self,
        index: SharedIndex,
        top_k: int = 3,           # How many chunks to send to the LLM after reranking
        candidate_k: int = 10,    # How many chunks to retrieve before reranking
        n_variants: int = 3,      # How many query rewrites to generate
        model_name: str = "llama-3.3-70b-versatile",
        reranker: CrossEncoder | None = None,
    ):
        """
        Args:
            index:        The shared FAISS + BM25 index (built once, shared across systems).
            top_k:        Final number of chunks fed to the LLM. Fewer = less noise.
            candidate_k:  Retrieval pool size before reranking. Bigger = more recall.
            n_variants:   Number of query rewrites to generate. More = better recall,
                          but more LLM tokens spent on rewriting.
            model_name:   Groq model used for both query rewriting and answer generation.
            reranker:     Optional pre-loaded CrossEncoder. Pass this when running
                          multiple systems in one script so the model isn't downloaded twice.
        """
        self._index = index
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.n_variants = n_variants
        self.model_name = model_name

        settings = get_settings()

        # LLM used for query rewriting — slightly higher temperature (0.4) so the
        # rephrased questions are diverse rather than near-identical copies.
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.4,
            max_tokens=256,
        )

        # LLM used for answer generation — low temperature (0.1) for factual,
        # consistent answers. More tokens allowed because answers can be detailed.
        self._llm_answer = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Load the cross-encoder reranker.
        # If one was passed in from outside (e.g. compare_systems.py loads it once
        # and shares it), reuse it — no point downloading the model twice.
        if reranker is not None:
            self._reranker = reranker
        else:
            print(f"Loading cross-encoder reranker ({_RERANKER_MODEL})...")
            self._reranker = CrossEncoder(_RERANKER_MODEL)

        # --- Prompt Templates ---

        # Prompt that asks the LLM to rephrase the user's question N different ways.
        # The output is a JSON array of strings which we parse in _parse_query_variants().
        self._rewrite_prompt = ChatPromptTemplate.from_template(
            f"Generate {n_variants} different ways to ask the following question. "
            "Use different wording but ask for the same information. "
            f"Return exactly {n_variants} variants as a JSON array of strings.\n\n"
            "Question: {question}\n\n"
            "JSON array:"
        )

        # Prompt that instructs the LLM to answer using ONLY the provided context.
        # This grounding instruction is key to reducing hallucinations.
        self._answer_prompt = ChatPromptTemplate.from_template(
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY the information in the context below. "
            "If the context does not contain enough information, say: "
            "'The document does not contain enough information to answer this question.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    async def query(self, qa_pair: QAPair) -> SystemOutput:
        """
        Full 4-stage pipeline: rewrite -> retrieve -> rerank -> answer.

        Args:
            qa_pair: Contains the user's question (and optionally a reference answer).

        Returns:
            SystemOutput with the generated answer, latency, and cost info.
        """
        start = time.time()
        chunks = self._index.chunks

        # Build a lookup table: chunk text -> its position (index) in the chunks list.
        # FAISS returns Document objects, not indices, so we need this to convert back.
        content_to_idx = {chunk.page_content: i for i, chunk in enumerate(chunks)}

        # ── STAGE 1: Query Rewriting ──────────────────────────────────────────
        # Ask the LLM to generate N alternative phrasings of the question.
        # This broadens the vocabulary we search with, improving recall.

        rw_messages = await self._rewrite_prompt.ainvoke({"question": qa_pair.question})
        rw_response = await self._llm.ainvoke(rw_messages)

        # Parse the JSON array of rewritten questions (falls back to [] on error).
        variants = _parse_query_variants(rw_response.content, qa_pair.question)

        # Combine the original question with all its variants.
        # Example: ["What is RAG?", "How does RAG work?", "Explain RAG", ...]
        all_queries = [qa_pair.question] + variants

        # ── STAGE 2: Hybrid Retrieval ─────────────────────────────────────────
        # For every query variant, run both BM25 and FAISS searches.
        # Then merge all the ranked result lists using Reciprocal Rank Fusion.
        #
        # We run this in a thread executor because both BM25 and FAISS are
        # CPU-bound / synchronous operations — running them in the async event
        # loop directly would block other coroutines.

        def _hybrid_retrieve_all():
            # Collect ranked result lists from both search methods
            bm25_rankings = []   # List of lists: each inner list = chunk indices in BM25 rank order
            faiss_rankings = []  # Same for FAISS

            for query in all_queries:
                # BM25 search — returns (score, text, chunk_index) tuples
                bm25_results = self._index.bm25.retrieve(query, k=self.candidate_k)
                bm25_rankings.append([idx for _, _, idx in bm25_results])

                # FAISS semantic search — returns Document objects
                faiss_docs = self._index.vectorstore.similarity_search(query, k=self.candidate_k)
                # Convert Document objects back to chunk indices using our lookup table.
                # Skip any docs not found in the table (shouldn't happen, but -1 is the sentinel).
                faiss_ranking = [content_to_idx.get(doc.page_content, -1) for doc in faiss_docs]
                faiss_rankings.append([i for i in faiss_ranking if i >= 0])

            # Combine ALL ranked lists (BM25 + FAISS from every query variant)
            # and merge them with Reciprocal Rank Fusion.
            # RRF gives higher combined scores to chunks that rank highly across
            # multiple lists, naturally surfacing the most consistently relevant chunks.
            all_rankings = bm25_rankings + faiss_rankings
            merged_indices = reciprocal_rank_fusion(all_rankings)

            # Return the top candidate_k chunks as Document objects
            return [chunks[i] for i in merged_indices[:self.candidate_k]]

        candidates = await asyncio.get_event_loop().run_in_executor(
            None, _hybrid_retrieve_all
        )

        # ── STAGE 3: Cross-Encoder Reranking ─────────────────────────────────
        # The cross-encoder reads the question AND each candidate chunk together
        # in a single forward pass, giving it much more context than the retrieval
        # step (which embeds question and chunks separately).
        #
        # Result: a relevance score per candidate. We keep only the top_k.
        # This is also CPU-bound, so we run it in a thread executor.

        def _rerank(candidates):
            # Build (question, chunk_text) pairs for the cross-encoder to score
            pairs = [(qa_pair.question, doc.page_content) for doc in candidates]

            # Cross-encoder returns a score for each pair (higher = more relevant)
            scores = self._reranker.predict(pairs)

            # Sort by score descending and keep only the top_k chunks
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[:self.top_k]]

        source_docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _rerank(candidates)
        )

        # ── STAGE 4: Answer Generation ────────────────────────────────────────
        # Join the top-k reranked chunks into one context block.
        # The "---" separator makes it clear to the LLM where each chunk ends.
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        messages = await self._answer_prompt.ainvoke(
            {"context": context, "question": qa_pair.question}
        )
        response = await self._llm_answer.ainvoke(messages)
        answer = response.content

        # ── Bookkeeping ───────────────────────────────────────────────────────
        latency_ms = (time.time() - start) * 1000

        # Store the context back on the QAPair so evaluators can use it
        # (e.g. faithfulness checks that the answer is grounded in this context)
        if context:
            qa_pair.context = context

        # Rough cost estimate: word count * 1.3 approximates token count,
        # then multiply by the per-token price.
        rw_tokens = len(rw_response.content.split()) * 1.3   # tokens used for query rewriting
        ans_tokens = len(answer.split()) * 1.3                # tokens used for the answer
        estimated_cost = (rw_tokens + ans_tokens) * _COST_PER_OUTPUT_TOKEN

        logger.info(
            f"AdvancedRAG answered in {latency_ms:.0f}ms | "
            f"{len(all_queries)} queries -> {len(candidates)} candidates -> "
            f"{len(source_docs)} after reranking"
        )

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "system": "advanced_rag",
                "n_query_variants": len(variants),          # how many rewrites were generated
                "candidates_before_rerank": len(candidates), # pool size going into reranker
                "chunks_after_rerank": len(source_docs),    # what the LLM actually sees
            },
        )
