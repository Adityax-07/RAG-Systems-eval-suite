"""
HyDE RAG System — Hypothetical Document Embeddings

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)

The Problem HyDE Solves:
  When you embed a *question* and a *passage that answers it*, their vectors
  are actually not that close — because questions and answers have very different
  linguistic structure.

  Example:
    Question: "What is FAISS?"
      -> embedding reflects the structure of a question
    Answer chunk: "FAISS (Facebook AI Similarity Search) is a library for efficient..."
      -> embedding reflects the structure of a factual statement

  These two vectors may not be very similar even though one directly answers
  the other. This is a fundamental limitation of embedding questions directly.

The HyDE Trick:
  Instead of embedding the raw question, we:
    1. Ask the LLM to write a SHORT, HYPOTHETICAL answer to the question
       (even if it might not be perfectly accurate — it just needs to be
       in the right "style" of a factual answer)
    2. Embed THAT hypothetical answer instead of the question
    3. Use the hypothetical answer's embedding to search FAISS

  Why does this work?
  The hypothetical answer is in the same "linguistic space" as real answer
  chunks in the document. So its embedding is much closer to the real answer
  chunks than the question's embedding would be.

  The key: we ONLY use the hypothesis for retrieval. The actual answer
  generation step still uses the REAL retrieved chunks, not the hypothesis.

Pipeline:
  Question
    -> LLM generates a hypothetical answer (just for embedding, not the final answer)
    -> Embed the hypothesis (not the question)
    -> FAISS searches for chunks closest to the hypothesis embedding
    -> LLM generates the real answer using the retrieved chunks
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


class HyDERAGSystem:
    """
    Retrieves chunks by embedding a hypothetical answer rather than the question.

    This costs one extra LLM call (to generate the hypothesis) but can
    significantly improve retrieval quality for questions that use very
    different vocabulary from the document.
    """

    def __init__(
        self,
        index: SharedIndex,
        top_k: int = 3,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        """
        Args:
            index:      The shared FAISS + BM25 index.
            top_k:      How many chunks to retrieve and send to the LLM.
            model_name: Groq model used for BOTH hypothesis generation and answer generation.
        """
        self._index = index
        self.top_k = top_k
        self.model_name = model_name

        settings = get_settings()

        # LLM for generating the hypothetical answer.
        # Slightly higher temperature (0.3) so the hypothesis is somewhat diverse
        # and covers different angles of the answer.
        # Short max_tokens (256) because we just need enough text to get a good embedding.
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=256,
        )

        # LLM for generating the final answer from real retrieved context.
        # Low temperature (0.1) for factual, consistent answers.
        self._llm_answer = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Prompt 1: Generate a HYPOTHETICAL answer for the embedding step only.
        # We tell the LLM "it's okay if you're not sure" because we don't actually
        # use this answer — we just need it to sound like a real answer passage.
        self._hypothesis_prompt = ChatPromptTemplate.from_template(
            "Write a short, factual paragraph (2-4 sentences) that would answer "
            "the following question. It's okay if you're not completely sure — "
            "write what a knowledgeable person would likely say.\n\n"
            "Question: {question}\n\n"
            "Paragraph:"
        )

        # Prompt 2: Generate the REAL answer using retrieved chunks.
        # Same grounding instruction as other RAG systems.
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
        HyDE pipeline: generate hypothesis -> embed it -> retrieve -> generate real answer.

        Args:
            qa_pair: Contains the question to answer.

        Returns:
            SystemOutput with the answer, context, timing, and cost.
        """
        start = time.time()

        # --- Step 1: Generate a Hypothetical Answer ---
        # This is an LLM call just to get a plausible-sounding passage.
        # We don't show this hypothesis to the user — it's purely internal.
        hyp_messages = await self._hypothesis_prompt.ainvoke({"question": qa_pair.question})
        hyp_response = await self._llm.ainvoke(hyp_messages)
        hypothesis = hyp_response.content
        logger.debug(f"HyDE hypothesis: {hypothesis[:100]}...")

        # --- Step 2: Retrieve Using the Hypothesis Embedding ---
        # We embed the hypothesis (not the original question) and use that
        # vector to find the most similar chunks in the FAISS index.
        # This runs in a thread because embedding + FAISS search are synchronous.
        def _retrieve_by_hypothesis():
            # Convert the hypothesis text into a vector using the same embedding
            # model that was used to build the FAISS index
            hyp_embedding = self._index.embeddings.embed_query(hypothesis)

            # Search FAISS using the hypothesis vector instead of the question vector
            return self._index.vectorstore.similarity_search_by_vector(
                hyp_embedding, k=self.top_k
            )

        source_docs = await asyncio.get_event_loop().run_in_executor(
            None, _retrieve_by_hypothesis
        )

        # --- Step 3: Generate the Real Answer from Retrieved Chunks ---
        # Now we use the actual retrieved chunks (not the hypothesis) to answer.
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

        # Cost = hypothesis LLM call + answer LLM call (two separate API calls)
        hyp_tokens = len(hypothesis.split()) * 1.3
        ans_tokens = len(answer.split()) * 1.3
        estimated_cost = (hyp_tokens + ans_tokens) * _COST_PER_OUTPUT_TOKEN

        logger.info(f"HyDE answered in {latency_ms:.0f}ms | {len(source_docs)} chunks via hypothesis embedding")

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "system": "hyde_rag",
                "hypothesis_length": len(hypothesis),   # how long the hypothesis was
                "chunks_retrieved": len(source_docs),
            },
        )
