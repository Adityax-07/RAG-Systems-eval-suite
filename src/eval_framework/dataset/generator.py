"""Synthetic QA dataset generator.

Given a corpus of documents, this generates question-answer pairs for evaluation.

Why you need this:
- Real eval datasets are expensive to collect manually
- Companies need to eval on THEIR data, not public benchmarks
- Synthetic data lets you bootstrap an eval suite in minutes

How it works (3-step pipeline):
  Step 1: CHUNK  — split documents into digestible passages
  Step 2: GENERATE — ask LLM to generate Q&A pairs from each chunk
  Step 3: VALIDATE — ask a second LLM call to verify the Q&A is good

Teaching note: This "generate then filter" pattern is used everywhere in ML
pipelines. Generate a lot cheaply, then filter expensively. You'll see it in
data augmentation, synthetic training data, and here in eval data generation.
"""

import asyncio
import logging
import uuid
import json
import re
from typing import Optional

from ..types import QAPair, DatasetExample, EvaluationMetric
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ─── Prompts ─────────────────────────────────────────────────────────────────
# Prompt engineering for dataset generation is an art.
# Key insight: ask for diverse question types to avoid a dataset that only
# has easy factoid questions. Real systems fail on reasoning and multi-hop queries.

QA_GENERATION_SYSTEM_PROMPT = """You are an expert at creating evaluation datasets for AI systems.

Given a passage of text, generate high-quality question-answer pairs for testing a RAG (Retrieval Augmented Generation) system.

Generate diverse question types:
- FACTOID: Direct fact lookup ("What is X?", "When did Y happen?")
- REASONING: Requires inference ("Why did X cause Y?", "What would happen if...?")
- MULTI-HOP: Requires combining multiple facts from the passage
- COMPARISON: Comparing entities or concepts ("How does X differ from Y?")

Rules:
1. Questions must be answerable ONLY from the passage (not general knowledge)
2. Answers must be grounded in specific text from the passage
3. Include the relevant quote from the passage as supporting evidence
4. Make questions challenging enough to test the system meaningfully
5. Vary difficulty: some easy, some medium, some hard

Respond with a JSON array of QA pairs."""

QA_GENERATION_USER_TEMPLATE = """Generate {n_questions} diverse question-answer pairs from this passage.

PASSAGE:
{passage}

Return a JSON array with this structure:
[
  {{
    "question": "...",
    "answer": "...",
    "question_type": "factoid|reasoning|multi_hop|comparison",
    "difficulty": "easy|medium|hard",
    "supporting_quote": "exact quote from passage",
    "tags": ["topic1", "topic2"]
  }}
]

Generate exactly {n_questions} pairs. Make them varied and challenging."""

QA_VALIDATION_SYSTEM_PROMPT = """You are a quality reviewer for AI evaluation datasets.

Review each question-answer pair and check:
1. Is the question clear and unambiguous?
2. Is the answer correct and grounded in the passage?
3. Can the question be answered from the passage alone?
4. Is the answer complete (not missing key information)?
5. Is it a meaningful test (not trivially easy)?

Be strict — only approve pairs that would genuinely test a RAG system."""

QA_VALIDATION_USER_TEMPLATE = """Review this question-answer pair for quality.

PASSAGE:
{passage}

QUESTION: {question}
ANSWER: {answer}

Is this a good evaluation pair? Respond with JSON:
{{
  "is_valid": true/false,
  "issues": ["list of any problems"],
  "quality_score": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""


# ─── Core classes ─────────────────────────────────────────────────────────────

class DatasetGenerator:
    """Generates synthetic evaluation datasets from document corpora.

    Usage:
        generator = DatasetGenerator(llm_client, questions_per_chunk=3)

        # From a list of documents
        dataset = await generator.generate_from_documents(
            documents=["doc text 1", "doc text 2", ...],
            n_questions_per_chunk=3,
        )

        print(f"Generated {len(dataset)} QA pairs")
        for ex in dataset[:3]:
            print(f"Q: {ex.qa_pair.question}")
            print(f"A: {ex.qa_pair.answer}")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        validate: bool = True,
        min_quality_score: float = 0.7,
        concurrency: int = 3,
    ):
        """
        Args:
            llm_client: LLM to use for generation and validation.
            chunk_size: Characters per chunk (800 ≈ 200 tokens, sweet spot for Q&A).
            chunk_overlap: Overlap between chunks to avoid cutting mid-sentence.
            validate: Whether to run a validation pass (recommended, costs more).
            min_quality_score: Threshold for keeping a QA pair after validation.
            concurrency: Max concurrent LLM calls (keep low, generation is expensive).
        """
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.validate = validate
        self.min_quality_score = min_quality_score
        self._semaphore = asyncio.Semaphore(concurrency)

    def chunk_document(self, text: str) -> list[str]:
        """Split a document into overlapping chunks.

        We use character-level chunking with overlap so context isn't lost
        at chunk boundaries. In production, you'd use semantic chunking
        (split on paragraph/section boundaries), but this is good enough.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            # Try to end at a sentence boundary for cleaner chunks
            if end < len(text):
                last_period = chunk.rfind(". ")
                if last_period > self.chunk_size // 2:
                    chunk = chunk[: last_period + 1]
            chunks.append(chunk.strip())
            start += len(chunk) - self.chunk_overlap
        return [c for c in chunks if len(c) > 100]  # Skip tiny chunks

    async def _generate_qa_for_chunk(
        self,
        chunk: str,
        n_questions: int = 3,
    ) -> list[dict]:
        """Generate raw QA pairs for a single chunk."""
        async with self._semaphore:
            user_prompt = QA_GENERATION_USER_TEMPLATE.format(
                passage=chunk,
                n_questions=n_questions,
            )
            try:
                response = await self.llm_client.generate(
                    system_prompt=QA_GENERATION_SYSTEM_PROMPT,
                    user_message=user_prompt,
                    temperature=0.7,  # Higher temp = more diverse questions
                    max_tokens=2000,
                )
                # Extract JSON array from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return json.loads(response)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"QA generation failed for chunk: {e}")
                return []

    async def _validate_qa_pair(
        self,
        chunk: str,
        question: str,
        answer: str,
    ) -> tuple[bool, float]:
        """Validate a QA pair — returns (is_valid, quality_score)."""
        async with self._semaphore:
            user_prompt = QA_VALIDATION_USER_TEMPLATE.format(
                passage=chunk,
                question=question,
                answer=answer,
            )
            try:
                response = await self.llm_client.generate(
                    system_prompt=QA_VALIDATION_SYSTEM_PROMPT,
                    user_message=user_prompt,
                    temperature=0.1,  # Low temp for consistent validation
                    max_tokens=300,
                )
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    is_valid = data.get("is_valid", False)
                    quality = float(data.get("quality_score", 0.5))
                    return is_valid and quality >= self.min_quality_score, quality
                return False, 0.0
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                return False, 0.0

    async def generate_from_documents(
        self,
        documents: list[str],
        n_questions_per_chunk: int = 3,
    ) -> list[DatasetExample]:
        """Main entry point: generate a dataset from a list of document strings.

        Args:
            documents: Raw document texts (e.g., loaded from PDFs, web pages).
            n_questions_per_chunk: How many Q&A pairs to generate per chunk.

        Returns:
            List of DatasetExample objects ready for evaluation.
        """
        # Step 1: Chunk all documents
        all_chunks: list[str] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"Chunked document into {len(chunks)} chunks")

        logger.info(
            f"Total: {len(all_chunks)} chunks across {len(documents)} documents. "
            f"Generating {n_questions_per_chunk} Q&A per chunk = "
            f"~{len(all_chunks) * n_questions_per_chunk} pairs before filtering."
        )

        # Step 2: Generate QA pairs for all chunks concurrently
        generation_tasks = [
            self._generate_qa_for_chunk(chunk, n_questions_per_chunk)
            for chunk in all_chunks
        ]
        chunk_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        # Step 3: Validate and convert to DatasetExample
        dataset_examples: list[DatasetExample] = []

        for chunk, raw_pairs in zip(all_chunks, chunk_results):
            if isinstance(raw_pairs, Exception):
                logger.warning(f"Chunk generation failed: {raw_pairs}")
                continue

            for raw_qa in raw_pairs:
                question = raw_qa.get("question", "")
                answer = raw_qa.get("answer", "")

                if not question or not answer:
                    continue

                # Validate if requested
                if self.validate:
                    is_valid, quality_score = await self._validate_qa_pair(
                        chunk, question, answer
                    )
                    if not is_valid:
                        logger.debug(
                            f"Filtered low-quality pair (score={quality_score:.2f}): {question[:50]}..."
                        )
                        continue
                else:
                    quality_score = 1.0

                example = DatasetExample(
                    id=str(uuid.uuid4()),
                    qa_pair=QAPair(
                        question=question,
                        answer=answer,
                        context=chunk,
                    ),
                    source_document=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    difficulty=raw_qa.get("difficulty", "medium"),
                    tags=raw_qa.get("tags", []) + [raw_qa.get("question_type", "factoid")],
                )
                dataset_examples.append(example)

        logger.info(
            f"Generated {len(dataset_examples)} validated QA pairs "
            f"(filtered {len(all_chunks) * n_questions_per_chunk - len(dataset_examples)} low-quality pairs)"
        )
        return dataset_examples

    async def generate_from_text(
        self,
        text: str,
        n_questions: int = 10,
    ) -> list[DatasetExample]:
        """Convenience method: generate from a single text string."""
        return await self.generate_from_documents(
            documents=[text],
            n_questions_per_chunk=max(1, n_questions // max(1, len(self.chunk_document(text)))),
        )
