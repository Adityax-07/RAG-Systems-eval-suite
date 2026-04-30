"""
Adaptive RAG System — The LLM Decides Its Own Retrieval Strategy

The Problem with Fixed Pipelines:
  Every other system in this benchmark uses the SAME retrieval strategy
  for every question, regardless of what the question actually needs.

  - Naive RAG uses FAISS for a simple factual question ("What is RLHF?")
    AND a complex multi-part question ("Compare FAISS and Pinecone across
    5 dimensions") — even though these need very different amounts of context.
  - Advanced RAG runs the full pipeline (rewrite + hybrid + rerank) even for
    a trivial question where a single chunk would have been enough.

The Fix — Route First, Retrieve Second:
  Before doing any retrieval, we ask the LLM to classify the query into
  one of three categories:

    "simple"    → A focused factual question. One clear answer exists in the
                  knowledge base. Naive RAG (fast FAISS search) is enough.
                  Example: "What is LoRA?" / "Who introduced the Transformer?"

    "complex"   → A multi-part, comparative, or reasoning-heavy question.
                  Needs broad, high-quality context. Advanced RAG is used:
                  query rewriting + hybrid search + cross-encoder reranking.
                  Example: "Compare BM25 and FAISS across recall, precision,
                  and latency" / "What are the production challenges in LLMOps?"

    "ambiguous" → The question is vague, uses jargon, or could be interpreted
                  multiple ways. Hybrid RAG (BM25 + FAISS) gives balanced
                  coverage without the full cost of Advanced RAG.
                  Example: "How does attention work?" / "What's the best RAG?"

  After classifying, the system runs the appropriate retrieval pipeline and
  logs which strategy was chosen — so you can inspect routing decisions in
  the benchmark results.

Why is this better than always running Advanced RAG?
  - Simple questions get answers faster (lower latency) and cheaper (less cost)
  - Complex questions still get the full pipeline — no quality sacrifice
  - The routing itself is a form of reasoning — the LLM understands what
    kind of retrieval its own question requires

Trade-offs:
  - One extra LLM call per question (the classification step)
  - If classification is wrong, the wrong pipeline runs
    (mitigated by the fallback: ambiguous → hybrid, which is middle-ground)
  - Average cost is between naive-rag and advanced-rag, not the cheapest
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

# We reuse the existing system implementations rather than duplicating code.
# Adaptive RAG is a router ON TOP of the other systems — not a new retrieval
# method itself.
from eval_framework.systems.naive_rag import NaiveRAGSystem
from eval_framework.systems.hybrid_rag import HybridRAGSystem
from eval_framework.systems.advanced_rag import AdvancedRAGSystem

logger = logging.getLogger(__name__)

_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000

# The three routing categories the LLM can choose from.
# Kept lowercase so we can do a simple string match on the LLM response.
_ROUTE_SIMPLE    = "simple"
_ROUTE_COMPLEX   = "complex"
_ROUTE_AMBIGUOUS = "ambiguous"


class AdaptiveRAGSystem:
    """
    A meta-RAG system that routes each query to the right retrieval pipeline.

    Instead of applying the same strategy to every question, this system:
      1. Classifies the query complexity with one LLM call
      2. Delegates to the appropriate sub-system:
           simple    -> NaiveRAGSystem    (FAISS only, fast)
           complex   -> AdvancedRAGSystem (rewrite + hybrid + rerank, best quality)
           ambiguous -> HybridRAGSystem   (BM25 + FAISS, balanced)
    """

    def __init__(
        self,
        index: SharedIndex,
        model_name: str = "llama-3.3-70b-versatile",
        reranker: CrossEncoder | None = None,
    ):
        """
        Args:
            index:      The shared FAISS + BM25 index, passed down to sub-systems.
            model_name: Groq model used for both routing classification and answers.
            reranker:   Pre-loaded CrossEncoder shared from compare_systems.py.
                        Avoids downloading the model multiple times.
        """
        self.model_name = model_name
        self._index = index

        settings = get_settings()

        # ── Router LLM ─────────────────────────────────────────────────────────
        # Low temperature (0.0) for deterministic routing decisions.
        # We don't want the classifier to be creative — just consistent.
        self._router_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.0,
            max_tokens=10,   # We only need one word back: "simple", "complex", or "ambiguous"
        )

        # ── Routing Prompt ─────────────────────────────────────────────────────
        # The prompt is deliberately strict:
        #   - Exactly one word response
        #   - Three options spelled out with clear definitions
        #   - Examples given for each category to reduce misclassification
        self._router_prompt = ChatPromptTemplate.from_template(
            "Classify the following question into exactly ONE category.\n\n"
            "Categories:\n"
            "- simple: A direct factual question with one clear answer.\n"
            "  Examples: 'What is FAISS?', 'Who introduced the Transformer?'\n\n"
            "- complex: A multi-part, comparative, or reasoning-heavy question.\n"
            "  Examples: 'Compare BM25 and FAISS', 'What are the production challenges of LLMOps?'\n\n"
            "- ambiguous: A vague or open-ended question that could be interpreted multiple ways.\n"
            "  Examples: 'How does attention work?', 'What is the best RAG approach?'\n\n"
            "Respond with ONLY the category name (simple / complex / ambiguous).\n\n"
            "Question: {question}\n\n"
            "Category:"
        )

        # ── Sub-Systems ────────────────────────────────────────────────────────
        # Instantiate all three sub-systems upfront.
        # They share the same index and reranker — no duplicate downloads.
        self._naive    = NaiveRAGSystem(index=index, model_name=model_name)
        self._hybrid   = HybridRAGSystem(index=index, model_name=model_name)
        self._advanced = AdvancedRAGSystem(
            index=index,
            model_name=model_name,
            reranker=reranker,
        )

    async def _classify(self, question: str) -> str:
        """
        Ask the router LLM to classify the question.

        Returns one of: "simple", "complex", "ambiguous".
        Falls back to "ambiguous" if the response is unexpected — this is the
        safest middle-ground (hybrid retrieval) when we're unsure.
        """
        messages = await self._router_prompt.ainvoke({"question": question})
        response = await self._router_llm.ainvoke(messages)

        # Normalize: strip whitespace, lowercase, take first word only.
        # LLMs sometimes return "Simple." or "COMPLEX\n" — this handles that.
        raw = response.content.strip().lower().split()[0].rstrip(".,:")

        if raw in (_ROUTE_SIMPLE, _ROUTE_COMPLEX, _ROUTE_AMBIGUOUS):
            return raw

        # If we got something unexpected (e.g. "moderate", "medium"), fall back
        # to ambiguous — hybrid search is a safe default.
        logger.warning(f"Unexpected route '{raw}' for: '{question}' — defaulting to ambiguous")
        return _ROUTE_AMBIGUOUS

    async def query(self, qa_pair: QAPair) -> SystemOutput:
        """
        Route the question to the right sub-system and return its output.

        The metadata field logs:
          - which route was chosen
          - how long the routing classification itself took
        so you can inspect routing decisions in the benchmark results.
        """
        start = time.time()

        # ── Step 1: Classify ───────────────────────────────────────────────────
        route = await self._classify(qa_pair.question)
        route_latency_ms = (time.time() - start) * 1000

        logger.info(f"AdaptiveRAG routed '{qa_pair.question[:60]}...' → {route.upper()} "
                    f"(classification took {route_latency_ms:.0f}ms)")

        # ── Step 2: Delegate to sub-system ────────────────────────────────────
        # Each sub-system returns a SystemOutput with its own latency/cost.
        # We add the routing overhead on top of that.
        if route == _ROUTE_SIMPLE:
            result = await self._naive.query(qa_pair)
        elif route == _ROUTE_COMPLEX:
            result = await self._advanced.query(qa_pair)
        else:
            # ambiguous (and any unexpected fallback)
            result = await self._hybrid.query(qa_pair)

        # ── Step 3: Patch metadata ─────────────────────────────────────────────
        # Add routing info to the result so it shows up in the benchmark DB.
        # The sub-system's own latency is already in result.latency_ms —
        # we add the classification overhead separately so both are visible.
        total_latency_ms = (time.time() - start) * 1000
        routing_cost = len(qa_pair.question.split()) * 1.3 * _COST_PER_OUTPUT_TOKEN

        result.latency_ms   = total_latency_ms
        result.cost_usd     = result.cost_usd + routing_cost
        result.model        = self.model_name
        result.metadata     = {
            **result.metadata,
            "system":              "adaptive_rag",
            "route":               route,                      # which pipeline was chosen
            "route_latency_ms":    round(route_latency_ms, 1), # cost of classification step
        }

        return result
