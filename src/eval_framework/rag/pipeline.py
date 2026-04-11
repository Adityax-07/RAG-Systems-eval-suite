"""Real RAG pipeline using LangChain + FAISS + Groq.

This is what replaces mock_system() in the CLI. Instead of echoing back the
reference answer, this:
  1. Loads your document (PDF or .txt)
  2. Splits it into overlapping chunks
  3. Embeds chunks locally using sentence-transformers (all-MiniLM-L6-v2)
  4. Stores vectors in FAISS (in-memory, no server needed)
  5. On each question: retrieves top-k chunks → sends to Groq LLM → returns answer

The answer and retrieved context are both returned so the faithfulness evaluator
can check whether the answer is grounded in what was actually retrieved.
"""

import asyncio
import logging
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from eval_framework.config import get_settings
from eval_framework.types import QAPair, SystemOutput

logger = logging.getLogger(__name__)

# Token cost estimate for llama-3.3-70b on Groq
# (Groq free tier is actually free, but we estimate for the cost metric)
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000


class RAGPipeline:
    """
    Simple but real RAG system. Compatible with the EvaluationPipeline interface:
        async def query(self, qa_pair: QAPair) -> SystemOutput

    Usage:
        rag = RAGPipeline("data/my_doc.pdf").build()
        # Then pass rag.query as the system function:
        report = await pipeline.run(system=rag.query, dataset=..., system_name="rag-v1")
    """

    def __init__(
        self,
        doc_path: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        top_k: int = 3,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        """
        Args:
            doc_path:      Path to a .pdf or .txt file to use as the knowledge base.
            chunk_size:    Characters per chunk. Smaller = more precise retrieval,
                           larger = more context per chunk.
            chunk_overlap: Characters of overlap between adjacent chunks so
                           sentences don't get cut at boundaries.
            top_k:         How many chunks to retrieve per question.
            model_name:    Groq model to use for answer generation.
        """
        self.doc_path = Path(doc_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.model_name = model_name
        self._llm = None
        self._retriever = None
        self._prompt = None

    def build(self) -> "RAGPipeline":
        """
        Load the document, embed chunks, and build the retrieval chain.
        Call this once before evaluation starts.
        """
        if not self.doc_path.exists():
            raise FileNotFoundError(f"Document not found: {self.doc_path}")

        logger.info(f"Loading document: {self.doc_path}")

        # Step 1: Load document
        if self.doc_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(self.doc_path))
        else:
            loader = TextLoader(str(self.doc_path), encoding="utf-8")
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} page(s)")

        # Step 2: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        # Step 3: Embed locally (no API key needed, ~80MB model downloads once)
        print("Loading embedding model (downloads once ~80MB)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Step 4: Build FAISS vector store (in-memory, instant)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS index built")

        # Step 5: Build Groq LLM
        settings = get_settings()
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=self.model_name,
            temperature=0.1,  # Low temp = more factual, less creative
            max_tokens=512,
        )

        # Step 6: Store retriever for use in query()
        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # Step 7: RAG prompt — tells the LLM to stay grounded in context
        self._prompt = ChatPromptTemplate.from_template(
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY the information in the context below. "
            "If the context does not contain enough information, say: "
            "'The document does not contain enough information to answer this question.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        print(f"RAG pipeline ready ({len(chunks)} chunks indexed)")
        return self

    async def query(self, qa_pair: QAPair) -> SystemOutput:
        """
        Query the RAG system for a single QA pair.

        This is the function you pass as `system=` to EvaluationPipeline.run().
        It also sets qa_pair.context to the retrieved chunks so faithfulness
        and context_precision evaluators can check the answer against them.
        """
        if self._retriever is None:
            raise RuntimeError("Call .build() before querying the RAG pipeline.")

        start = time.time()

        # Step 1: Retrieve relevant chunks (sync call in thread pool)
        source_docs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._retriever.invoke(qa_pair.question),
        )

        # Step 2: Format context from retrieved chunks
        context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        # Step 3: Build prompt and call Groq LLM (async)
        messages = await self._prompt.ainvoke({"context": context, "question": qa_pair.question})
        response = await self._llm.ainvoke(messages)
        answer = response.content

        latency_ms = (time.time() - start) * 1000

        # Inject retrieved context into the qa_pair so evaluators can use it.
        # This is what makes faithfulness meaningful — we're checking the answer
        # against what was actually retrieved, not some pre-written context.
        if context:
            qa_pair.context = context

        # Rough cost estimate (output tokens × price)
        output_tokens = len(answer.split()) * 1.3
        estimated_cost = output_tokens * _COST_PER_OUTPUT_TOKEN

        logger.info(
            f"RAG answered in {latency_ms:.0f}ms | "
            f"{len(source_docs)} chunks retrieved | "
            f"answer length: {len(answer)} chars"
        )

        return SystemOutput(
            answer=answer,
            latency_ms=latency_ms,
            cost_usd=estimated_cost,
            model=self.model_name,
            metadata={
                "chunks_retrieved": len(source_docs),
                "source_snippets": [doc.page_content[:100] for doc in source_docs],
            },
        )
