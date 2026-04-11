"""
Shared Index — Built Once, Used by All RAG Systems

Why does this exist?
  Every RAG system in this benchmark (Hybrid, Reranking, HyDE, QueryRewriting,
  Advanced) needs to search the same knowledge base. Each search requires:
    - A FAISS vectorstore (for semantic/embedding-based search)
    - A BM25 index (for keyword-based search)
    - The embedding model that was used to build the FAISS index

  Building these from scratch for each system would mean:
    - Loading the ~80MB embedding model 6 separate times
    - Re-encoding all 35 chunks 6 separate times
  That's slow and wasteful.

  Instead, we build everything ONCE here and share it across all systems.

What is FAISS?
  Facebook AI Similarity Search — a library for fast vector search.
  We store each text chunk as a vector (list of numbers capturing its meaning).
  When querying, we embed the question and find the chunks with the closest vectors.

What is BM25?
  Best Match 25 — a classic information retrieval algorithm based on word frequency.
  No neural network needed. It scores chunks based on how often the query words
  appear in them (with adjustments for document length and word rarity).
  Fast, interpretable, and great at exact keyword matches.

What is Reciprocal Rank Fusion (RRF)?
  A formula for merging multiple ranked lists into one combined ranking.

  Example: BM25 ranks chunk A at #1, FAISS ranks it at #3.
           BM25 ranks chunk B at #5, FAISS ranks it at #1.
  RRF combines these positions into a single score for each chunk.
  The formula: score = 1/(60 + rank)  — summed across all lists.
  The constant 60 prevents any single top-ranked result from dominating.
"""

import logging
from pathlib import Path

import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Rough cost per output token for Groq-hosted Llama models
_COST_PER_OUTPUT_TOKEN = 0.59 / 1_000_000


class BM25Index:
    """
    A simple wrapper around the rank_bm25 library to give it a consistent
    interface that matches how we call FAISS (retrieve method with k parameter).

    BM25 works by:
      1. Splitting every chunk into individual words (tokenization)
      2. When searching, splitting the query into words too
      3. Scoring each chunk based on how often query words appear in it,
         adjusted for document length (so longer docs don't get unfair advantage)
    """

    def __init__(self, chunks):
        """
        Build the BM25 index from a list of document chunks.

        Args:
            chunks: List of LangChain Document objects (the same chunks used by FAISS).
        """
        self.chunks = chunks

        # Tokenize: split each chunk's text into lowercase words
        # BM25 compares word overlap, so we normalize to lowercase
        tokenized = [doc.page_content.lower().split() for doc in chunks]

        # Build the BM25 index from the tokenized chunks
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> list:
        """
        Find the top-k chunks most relevant to the query by keyword overlap.

        Args:
            query: The search query (will be tokenized the same way as chunks).
            k:     Number of top results to return.

        Returns:
            List of (Document, score, original_index) tuples, sorted by score descending.
            The original_index lets callers map results back to the full chunks list.
        """
        # Tokenize the query the same way we tokenized the chunks
        tokens = query.lower().split()

        # Get a BM25 relevance score for each chunk
        scores = self._bm25.get_scores(tokens)

        # argsort gives indices that would sort the array ascending,
        # [::-1] reverses it to descending (highest score first), [:k] takes top-k
        top_indices = np.argsort(scores)[::-1][:k]

        return [(self.chunks[i], float(scores[i]), i) for i in top_indices]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """
    Combine multiple ranked lists of chunk indices into one merged ranking.

    This is the "glue" that makes Hybrid RAG, Query Rewriting, and Advanced RAG
    work — it takes rankings from different search methods/queries and produces
    one unified ranking.

    How it works:
      Each chunk in each ranked list gets a score of: 1 / (k + rank)
      where rank is its position in that list (0-based).
      These scores are summed across all lists.
      Chunks that rank well in multiple lists accumulate higher total scores.

    Why k=60?
      The constant k prevents the top-ranked item from scoring infinitely higher
      than the second. It's an empirically validated default from the original paper.

    Example:
      BM25  ranking: [3, 7, 12, ...]
      FAISS ranking: [7, 3, 21, ...]

      Chunk 3:  1/(60+0) + 1/(60+1) = 0.01667 + 0.01639 = 0.03306
      Chunk 7:  1/(60+1) + 1/(60+0) = 0.01639 + 0.01667 = 0.03306   (same — tied!)
      Chunk 12: 1/(60+2)             = 0.01613                        (only in BM25)
      Chunk 21: 1/(60+2)             = 0.01613                        (only in FAISS)

    Args:
        rankings: List of ranked lists. Each inner list contains chunk indices
                  sorted from most to least relevant according to one search method.
        k:        RRF smoothing constant (default 60 is standard).

    Returns:
        Single list of chunk indices sorted by combined RRF score (best first).
    """
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, chunk_idx in enumerate(ranking):
            # Add this chunk's RRF contribution for this ranked list
            # rank+1 because RRF is conventionally 1-based (rank=0 -> position 1)
            scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)

    # Sort chunks by their total accumulated RRF score, highest first
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class SharedIndex:
    """
    Loads a document, splits it into chunks, and builds both FAISS and BM25 indexes.

    Build this ONCE and pass it to every RAG system. All systems will search
    the exact same index, ensuring a fair comparison.

    Usage:
        index = SharedIndex("data/knowledge_base.txt").build()

        naive_rag   = NaiveRAGSystem(index, ...)
        hybrid_rag  = HybridRAGSystem(index, ...)
        advanced_rag = AdvancedRAGSystem(index, ...)
    """

    def __init__(
        self,
        doc_path: str | Path,
        chunk_size: int = 500,     # Each chunk is at most 500 characters
        chunk_overlap: int = 75,   # Chunks overlap by 75 chars so context isn't cut off at boundaries
    ):
        """
        Args:
            doc_path:      Path to the knowledge base file (.txt or .pdf).
            chunk_size:    Maximum size of each text chunk in characters.
                           Smaller = more precise retrieval, larger = more context per chunk.
            chunk_overlap: How many characters each chunk shares with the previous one.
                           Prevents important sentences from being split across chunks.
        """
        self.doc_path = Path(doc_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # These are populated when build() is called
        self.chunks = []
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vectorstore: FAISS | None = None
        self.bm25: BM25Index | None = None

    def build(self) -> "SharedIndex":
        """
        Load the document, split it, embed it, and build both indexes.

        Returns:
            self — so you can chain: index = SharedIndex(...).build()
        """
        if not self.doc_path.exists():
            raise FileNotFoundError(f"Document not found: {self.doc_path}")

        # --- Step 1: Load the document ---
        # Support both PDF (multi-page) and plain text files
        if self.doc_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(self.doc_path))
        else:
            loader = TextLoader(str(self.doc_path), encoding="utf-8")

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} page(s) from {self.doc_path}")

        # --- Step 2: Split into chunks ---
        # RecursiveCharacterTextSplitter tries to split on paragraph breaks (\n\n)
        # first, then line breaks, then sentences, etc. — preserving natural boundaries.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(self.chunks)} chunks")

        # --- Step 3: Build the FAISS vectorstore ---
        # This downloads the embedding model (~80MB) on first run, then caches it.
        # The model converts text chunks into 384-dimensional vectors.
        print("Loading embedding model (downloads once ~80MB)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # normalize for cosine similarity
        )

        # Embed all chunks and store them in a FAISS index for fast vector search
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        logger.info("FAISS index built")

        # --- Step 4: Build the BM25 index ---
        # BM25 doesn't need embeddings — it works directly on the text
        self.bm25 = BM25Index(self.chunks)
        logger.info("BM25 index built")

        print(f"Shared index ready — {len(self.chunks)} chunks, FAISS + BM25")
        return self
