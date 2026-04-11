"""RAG pipeline module — wraps LangChain + FAISS + Groq into an eval-compatible system."""

from .pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
