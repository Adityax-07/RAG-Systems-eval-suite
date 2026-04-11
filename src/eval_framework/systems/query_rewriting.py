"""
Query Rewriting RAG System — Multi-Query Retrieval

The Problem This Solves:
  A single phrasing of a question may miss relevant chunks that happen to use
  different vocabulary. The document and the question might be talking about
  the same thing but using different words.

  Example: If you ask "What is FAISS?" but the document talks about
  "vector similarity search with Facebook AI Similarity Search", a naive
  search might not surface those chunks because the words don't match well.

The Fix — Ask the Same Question Multiple Ways:
  Instead of searching once, we:
    1. Ask the LLM to rephrase the question N different ways
    2. Run a FAISS search for EACH rephrased version
    3. Merge all the result lists using RRF (Reciprocal Rank Fusion)
    4. Use the merged top-k chunks as context for the final answer

  Example with N=3:
    Original:  "What is FAISS?"
    Variant 1: "How does Facebook AI Similarity Search work?"
    Variant 2: "Explain vector database indexing with FAISS"
    Variant 3: "What types of indexes does FAISS support?"

  Each variant may retrieve DIFFERENT chunks. The merge step ensures that
  chunks appearing across multiple search results rank highest.

This is different from HyDE:
  - HyDE changes WHAT we embed (a hypothetical answer instead of the question)
  - Query Rewriting changes HOW MANY times we search (N queries instead of 1)
  These are complementary — Advanced RAG combines both with Hybrid search.

Helper Function — _parse_query_variants():
  The LLM returns the variants as a JSON array like: ["Q1", "Q2", "Q3"]
  This function parses that output robustly, with a fallback for when the
  LLM doesn't return valid JSON (it tries to extract numbered lines instead).
"""

import asyncio
import json
import logging
import re
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from eval_framework.config import get_settings
from eval_framework.systems.shared import SharedIndex, reciprocal_rank_fusion
from eval_framework.types import QAPair, SystemOutput

logger = logging.getLogger(__name__)

# Rough cost per output token for Groq-hosted Llama models
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000

# Default number of query variants to generate
_N_VARIANTS = 3


def _parse_query_variants(text: str, original: str) -> list[str]:
    """
    Extract a list of query variants from the LLM's raw response text.

    The LLM is asked to return a JSON array like: ["Q1", "Q2", "Q3"]
    But LLMs don't always follow formatting instructions perfectly, so
    we have two parsing strategies:

    Strategy 1 (preferred): Look for a JSON array [...] in the text and parse it.
    Strategy 2 (fallback):  Extract numbered/bulleted lines from the text.

    Args:
        text:     Raw text output from the LLM.
        original: The original question — returned as fallback if parsing fails completely.

    Returns:
        List of query variant strings.
    """
    # Strategy 1: Try to find and parse a JSON array in the response
    # re.DOTALL allows the pattern to match across multiple lines
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            variants = json.loads(match.group())
            # Make sure we got a list of strings (not numbers or nested objects)
            if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                return [v.strip() for v in variants if v.strip()]
        except (json.JSONDecodeError, ValueError):
            pass  # Fall through to strategy 2

    # Strategy 2: Extract lines that look like numbered/bulleted items
    # Strip leading "1.", "-", "*" etc. from each line
    lines = [
        re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("{")
    ]
    # Only keep lines that are long enough to be a real question (> 10 chars)
    variants = [line for line in lines if len(line) > 10]

    # If we got something, return up to N_VARIANTS. Otherwise return the original.
    return variants[:_N_VARIANTS] if variants else [original]


class QueryRewritingRAGSystem:
    """
    Generates multiple phrasings of the question, retrieves for each one,
    then merges all results using Reciprocal Rank Fusion before answering.

    Expected improvement over Naive RAG: better recall for ambiguous or
    short queries, and for topics that have multiple common phrasings.
    """

    def __init__(
        self,
        index: SharedIndex,
        top_k: int = 3,
        n_variants: int = _N_VARIANTS,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        """
        Args:
            index:      The shared FAISS + BM25 index.
            top_k:      Final number of chunks passed to the LLM.
            n_variants: How many query rewrites to generate. More = better recall,
                        but costs more tokens on the rewriting LLM call.
            model_name: Groq model used for both query rewriting and answer generation.
        """
        self._index = index
        self.top_k = top_k
        self.n_variants = n_variants
        self.model_name = model_name

        settings = get_settings()

        # LLM for generating query variants.
        # Higher temperature (0.4) encourages diverse, different-sounding rewrites.
        # If temperature is too low, variants end up nearly identical.
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.4,
            max_tokens=256,
        )

        # LLM for generating the final answer.
        # Low temperature (0.1) for factual consistency.
        self._llm_answer = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Prompt asking the LLM to rephrase the question N times as a JSON array
        self._rewrite_prompt = ChatPromptTemplate.from_template(
            f"Generate {n_variants} different ways to ask the following question. "
            "Each variant should use different wording but ask for the same information. "
            f"Return exactly {n_variants} variants as a JSON array of strings.\n\n"
            "Question: {question}\n\n"
            "JSON array:"
        )

        # Same grounding prompt used by all other RAG systems
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
        Query rewriting pipeline: rewrite -> retrieve each -> RRF merge -> answer.

        Args:
            qa_pair: Contains the question to answer.

        Returns:
            SystemOutput with the answer, context, timing, and cost.
        """
        start = time.time()
        chunks = self._index.chunks

        # Build a lookup: chunk text -> its index in the full chunks list
        # Needed because FAISS returns Document objects (not indices)
        content_to_idx = {chunk.page_content: i for i, chunk in enumerate(chunks)}

        # --- Step 1: Generate Query Variants ---
        # Ask the LLM for N rephrased versions of the question
        rw_messages = await self._rewrite_prompt.ainvoke({"question": qa_pair.question})
        rw_response = await self._llm.ainvoke(rw_messages)

        # Parse the JSON array from the LLM response
        variants = _parse_query_variants(rw_response.content, qa_pair.question)

        # Combine original question with all its variants
        all_queries = [qa_pair.question] + variants
        logger.debug(f"Query variants: {all_queries}")

        # --- Step 2: Retrieve for Each Query ---
        # For each query variant, run FAISS search and collect a ranked list.
        # This is CPU-bound so we run it in a thread to avoid blocking.
        def _retrieve_all():
            rankings = []
            for query in all_queries:
                # FAISS search for this specific query variant
                docs = self._index.vectorstore.similarity_search(query, k=self.top_k + 2)
                # Convert Document objects to chunk indices
                ranking = [content_to_idx.get(doc.page_content, -1) for doc in docs]
                # Filter out any -1s (safety guard)
                rankings.append([i for i in ranking if i >= 0])
            return rankings
            # rankings = [[3, 7, 12], [7, 3, 21], [12, 7, 3], ...]
            # Each inner list = chunk indices returned by one query variant

        rankings = await asyncio.get_event_loop().run_in_executor(None, _retrieve_all)

        # --- Step 3: Merge All Rankings with RRF ---
        # Chunks that appear highly ranked across multiple query variants
        # will have the highest combined RRF score.
        merged_indices = reciprocal_rank_fusion(rankings)
        source_docs = [chunks[i] for i in merged_indices[:self.top_k]]

        # --- Step 4: Generate Final Answer ---
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)
        messages = await self._answer_prompt.ainvoke(
            {"context": context, "question": qa_pair.question}
        )
        response = await self._llm_answer.ainvoke(messages)
        answer = response.content

        latency_ms = (time.time() - start) * 1000

        # Store context for the evaluators
        if context:
            qa_pair.context = context

        # Cost = query rewriting LLM call + answer LLM call
        rw_tokens = len(rw_response.content.split()) * 1.3
        ans_tokens = len(answer.split()) * 1.3
        estimated_cost = (rw_tokens + ans_tokens) * _COST_PER_OUTPUT_TOKEN

        logger.info(
            f"QueryRewriting answered in {latency_ms:.0f}ms | "
            f"{len(all_queries)} queries -> {len(source_docs)} unique chunks"
        )

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "system": "query_rewriting_rag",
                "n_variants": len(variants),           # how many rewrites were generated
                "chunks_retrieved": len(source_docs),  # final chunks sent to the LLM
            },
        )
