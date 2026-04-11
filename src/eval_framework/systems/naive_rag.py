"""
Naive RAG System — Basic Retrieval-Augmented Generation

This is the simplest form of RAG (Retrieval-Augmented Generation).
It adds one key improvement over the Base LLM: before answering, it
searches a knowledge base to find relevant text chunks, then hands
those chunks to the LLM as context.

How it works:
  Step 1 — Embed the question: Convert the question into a vector (list of numbers)
            that captures its meaning.
  Step 2 — FAISS search: Find the chunks in the knowledge base whose vectors
            are closest to the question vector (semantic similarity).
  Step 3 — Generate: Give the top-k chunks + the question to the LLM and ask
            it to answer using ONLY what's in those chunks.

Why is this "naive"?
  - It only uses one search method (vector similarity)
  - It searches with the raw question as-is (no rephrasing)
  - It uses all retrieved chunks equally (no reranking by relevance)

  Later systems fix these limitations one by one.

This is the "entry-level RAG" baseline. Every advanced system should beat it.
"""

import asyncio
import logging
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from eval_framework.config import get_settings
from eval_framework.systems.shared import SharedIndex
from eval_framework.types import QAPair, SystemOutput

logger = logging.getLogger(__name__)

# Rough cost per output token for Groq-hosted Llama models
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000


class NaiveRAGSystem:
    """
    Standard RAG: embed question -> FAISS similarity search -> LLM generation.

    The starting point that every advanced technique builds on top of.
    Should score significantly better than BaseLLM on faithfulness and
    hallucination rate, since the LLM is grounded by real retrieved context.
    """

    def __init__(
        self,
        index: SharedIndex,
        top_k: int = 3,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        """
        Args:
            index:      The shared FAISS + BM25 index (built once, reused here).
            top_k:      How many chunks to retrieve and show to the LLM.
                        3 is a good balance — enough context, not too noisy.
            model_name: Groq model for answer generation.
        """
        self._index = index
        self.top_k = top_k
        self.model_name = model_name

        settings = get_settings()

        # LLM for generating answers — low temperature for factual consistency
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Set up FAISS as a LangChain retriever.
        # "similarity" means: embed the query and find the nearest chunks by cosine distance.
        self._retriever = index.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        # Prompt instructs the LLM to ONLY use the provided context.
        # This "grounding" instruction is what reduces hallucinations in RAG systems.
        self._prompt = ChatPromptTemplate.from_template(
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
        Retrieve relevant chunks with FAISS, then generate a grounded answer.

        Args:
            qa_pair: Contains the question to answer.

        Returns:
            SystemOutput with the answer, retrieved context, timing, and cost.
        """
        start = time.time()

        # FAISS similarity search is a synchronous (blocking) operation.
        # We run it in a thread executor so it doesn't block the async event loop
        # while other questions are being processed concurrently.
        source_docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._retriever.invoke(qa_pair.question)
        )

        # Join the retrieved chunks into one context block.
        # The "---" separator helps the LLM see where each chunk starts and ends.
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        # Send context + question to the LLM for answer generation
        messages = await self._prompt.ainvoke({"context": context, "question": qa_pair.question})
        response = await self._llm.ainvoke(messages)
        answer = response.content

        latency_ms = (time.time() - start) * 1000

        # Store context on qa_pair so evaluators (faithfulness, etc.) can use it
        if context:
            qa_pair.context = context

        # Rough cost estimate: word count * 1.3 approximates token count
        output_tokens = len(answer.split()) * 1.3
        estimated_cost = output_tokens * _COST_PER_OUTPUT_TOKEN

        logger.info(f"NaiveRAG answered in {latency_ms:.0f}ms | {len(source_docs)} chunks")

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "system": "naive_rag",
                "chunks_retrieved": len(source_docs),
                # Store first 100 chars of each chunk for debugging
                "source_snippets": [doc.page_content[:100] for doc in source_docs],
            },
        )
