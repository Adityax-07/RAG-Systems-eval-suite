"""
Base LLM System — No Retrieval (The Simplest Possible System)

This is the "floor" of the benchmark — it represents what happens when you
ask an LLM a question WITHOUT giving it any documents to look at.

The LLM has to answer purely from what it learned during training (called
"parametric knowledge"). It has no access to your specific knowledge base.

Why do we include this?
  - It sets the baseline: every other system should score BETTER than this
  - It shows how much adding retrieval actually helps
  - Faithfulness will be low (there's no context to be faithful to)
  - Hallucination rate will be high (the model just makes things up if unsure)

Think of it like asking someone a question with no reference material vs.
letting them look up the answer in a book first.
"""

import logging
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from eval_framework.config import get_settings
from eval_framework.types import QAPair, SystemOutput

logger = logging.getLogger(__name__)

# Rough cost per output token for Groq-hosted Llama models
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000


class BaseLLMSystem:
    """
    Pure LLM with no retrieval — answers from training knowledge only.

    This is system #1 in the comparison. Every other system (Naive RAG,
    Hybrid RAG, etc.) adds retrieval on top of this foundation.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """
        Args:
            model_name: The Groq model to use for generating answers.
                        Same model is used across all systems for a fair comparison.
        """
        self.model_name = model_name
        settings = get_settings()

        # Set up the LLM client
        # Low temperature (0.1) = more deterministic, consistent answers
        # This is important for fair evaluation — we don't want scores to
        # vary just because the model got "creative" on one run
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=512,
        )

        # Simple prompt — just ask the question, no context provided
        # Notice there's no {context} variable here, unlike all other systems
        self._prompt = ChatPromptTemplate.from_template(
            "You are a knowledgeable assistant. "
            "Answer the following question as accurately as you can based on your training knowledge.\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    async def query(self, qa_pair: QAPair) -> SystemOutput:
        """
        Answer a question using only the LLM's built-in knowledge.

        Args:
            qa_pair: Contains the question to answer.

        Returns:
            SystemOutput with the answer, timing, and cost info.
        """
        start = time.time()

        # Format the prompt with the question and send to the LLM
        messages = await self._prompt.ainvoke({"question": qa_pair.question})
        response = await self._llm.ainvoke(messages)
        answer = response.content

        latency_ms = (time.time() - start) * 1000

        # Set context to empty string — this tells the evaluators there was
        # no retrieved context. The faithfulness evaluator will give a low score
        # because the answer can't be "grounded" in anything.
        qa_pair.context = ""

        # Rough cost estimate: word count * 1.3 approximates token count
        output_tokens = len(answer.split()) * 1.3
        estimated_cost = output_tokens * _COST_PER_OUTPUT_TOKEN

        logger.info(f"BaseLLM answered in {latency_ms:.0f}ms")

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "system": "base_llm",
                "chunks_retrieved": 0,   # No retrieval happened
            },
        )
