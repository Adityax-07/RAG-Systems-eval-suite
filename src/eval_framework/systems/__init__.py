"""RAG system variants for benchmarking.

Each system implements:
    async def query(self, qa_pair: QAPair) -> SystemOutput

Pass any system's .query method as the `system=` argument to EvaluationPipeline.run().

Systems (ordered by sophistication):
    BaseLLMSystem           — no retrieval, pure parametric knowledge
    NaiveRAGSystem          — FAISS semantic search only
    HybridRAGSystem         — BM25 + FAISS merged with RRF
    RerankingRAGSystem      — FAISS + cross-encoder reranker
    HyDERAGSystem           — hypothetical document embedding retrieval
    QueryRewritingRAGSystem — multi-query retrieval with RRF merge
    AdvancedRAGSystem       — hybrid + reranking + query rewriting (all combined)
    AdaptiveRAGSystem       — routes each query to the right pipeline dynamically
"""

from eval_framework.systems.shared import SharedIndex
from eval_framework.systems.base_llm import BaseLLMSystem
from eval_framework.systems.naive_rag import NaiveRAGSystem
from eval_framework.systems.hybrid_rag import HybridRAGSystem
from eval_framework.systems.reranking_rag import RerankingRAGSystem
from eval_framework.systems.hyde_rag import HyDERAGSystem
from eval_framework.systems.query_rewriting import QueryRewritingRAGSystem
from eval_framework.systems.advanced_rag import AdvancedRAGSystem
from eval_framework.systems.adaptive_rag import AdaptiveRAGSystem

__all__ = [
    "SharedIndex",
    "BaseLLMSystem",
    "NaiveRAGSystem",
    "HybridRAGSystem",
    "RerankingRAGSystem",
    "HyDERAGSystem",
    "QueryRewritingRAGSystem",
    "AdvancedRAGSystem",
    "AdaptiveRAGSystem",
]
