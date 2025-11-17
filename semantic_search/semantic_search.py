#!/usr/bin/env python3
"""
Semantic Search Engine for Twitter API Postman Documentation
Main CLI entry point with JSON output
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import (
    BASE_DIR,
    DATA_DIR,
    LOGS_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    DEFAULT_TOP_K,
    HYBRID_SEARCH_ENABLED,
    RERANKING_ENABLED,
    SEMANTIC_WEIGHT,
    DOCS_PATH,
    DOCS_REPO_URL,
    LOG_LEVEL,
    LOG_PATH,
    VECTOR_STORE_TYPE,
    METADATA_CACHE_PATH,
    EMBEDDINGS_CACHE_PATH,
    CHUNKS_CACHE_PATH,
    PINECONE_API_KEY
)

from utils.helpers import setup_logging, format_json_output, format_search_result
from ingestion.loader import DocumentationLoader
from ingestion.chunker import SemanticChunker
from embeddings.embedder import Embedder
from index.vector_store import PineconeVectorStore, LocalVectorStore
from search.retriever import HybridRetriever


# Setup logging
logger = setup_logging(LOG_PATH, LOG_LEVEL)


class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
        use_hybrid: bool = HYBRID_SEARCH_ENABLED,
        use_reranking: bool = RERANKING_ENABLED,
        semantic_weight: float = SEMANTIC_WEIGHT,
        vector_store_type: str = VECTOR_STORE_TYPE,
        metadata_path: Path = METADATA_CACHE_PATH,
        embeddings_path: Path = EMBEDDINGS_CACHE_PATH,
        chunks_path: Path = CHUNKS_CACHE_PATH
    ):
        """
        Initialize semantic search engine.

        Args:
            embedding_model: Model name for embeddings
            reranker_model: Model name for re-ranking
            use_hybrid: Whether to use hybrid search
            use_reranking: Whether to use re-ranking
            semantic_weight: Weight for semantic scores
            vector_store_type: Type of vector store ("pinecone")
            metadata_path: Path to metadata
            embeddings_path: Path to embeddings cache
            chunks_path: Path to chunks cache
        """
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.semantic_weight = semantic_weight
        self.vector_store_type = vector_store_type

        self.metadata_path = metadata_path
        self.embeddings_path = embeddings_path
        self.chunks_path = chunks_path

        self.retriever = None
        self.embedder = None
        self.vector_store = None
        self.chunks = []
    
    def _resolve_vector_store_type(self) -> str:
        """Return the effective vector store type, falling back when needed."""
        store_type = (self.vector_store_type or "").strip().lower()

        if store_type == "pinecone" and not PINECONE_API_KEY:
            logger.warning(
                "Pinecone API key not provided. Falling back to local vector store."
            )
            return "local"

        if store_type in {"pinecone", "local", "faiss", "flat"}:
            # Treat aliases like 'faiss' or 'flat' as the local implementation.
            return "pinecone" if store_type == "pinecone" else "local"

        logger.warning(
            "Unknown vector store type '%s'. Defaulting to local vector store.",
            store_type or "<empty>"
        )
        return "local"

    def _create_vector_store(self, embedding_dim: int):
        """Instantiate the configured vector store implementation."""
        store_type = self._resolve_vector_store_type()

        if store_type == "pinecone":
            return PineconeVectorStore(embedding_dim)

        return LocalVectorStore(embedding_dim)

    def build_index(
        self,
        clone_docs: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ) -> bool:
        """
        Build complete search index from documentation.
        
        Args:
            clone_docs: Whether to clone docs from GitHub
            chunk_size: Target chunk size in tokens
            chunk_overlap: Token overlap between chunks
            
        Returns:
            True if successful
        """
        try:
            logger.info("=" * 80)
            logger.info("BUILDING SEMANTIC SEARCH INDEX")
            logger.info("=" * 80)
            
            # 1. Load documentation
            logger.info(f"\n[1/5] Loading documentation from {DOCS_PATH}")
            loader = DocumentationLoader(DOCS_PATH, DOCS_REPO_URL)
            
            if clone_docs and not DOCS_PATH.exists():
                if not loader.clone_repo():
                    logger.error("Failed to clone documentation")
                    return False
            
            documents = loader.load_all()
            if not documents:
                logger.error("No documents loaded")
                return False
            
            # 2. Chunk documents
            logger.info(f"\n[2/5] Chunking {len(documents)} documents")
            chunker = SemanticChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.chunk_documents(documents)
            self.chunks = chunks
            
            if not chunks:
                logger.error("No chunks created")
                return False
            
            # 3. Generate embeddings
            logger.info(f"\n[3/5] Generating embeddings for {len(chunks)} chunks")
            self.embedder = Embedder(self.embedding_model)
            
            chunk_texts = [chunk.get("content", "") for chunk in chunks]
            embeddings = self.embedder.embed_texts(
                chunk_texts,
                batch_size=32,
                normalize=True,
                show_progress=True
            )
            
            # Save embeddings cache
            self.embedder.save_embeddings(embeddings, self.embeddings_path)
            
            # 4. Create vector index
            logger.info(f"\n[4/5] Creating {self._resolve_vector_store_type()} index")
            embedding_dim = self.embedder.get_embedding_dimension()
            self.vector_store = self._create_vector_store(embedding_dim)

            # Prepare metadata for each chunk
            metadata = [
                {
                    "chunk_id": i,
                    "source": chunk.get("source", "unknown"),
                    "section": chunk.get("section", ""),
                    "size": chunk.get("size", 0),
                    "content_preview": chunk.get("content", "")[:200]
                }
                for i, chunk in enumerate(chunks)
            ]

            self.vector_store.add_vectors(embeddings, metadata)
            self.vector_store.save(self.embeddings_path, self.metadata_path)
            
            # Save chunks for later loading
            self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(chunks, f)
            logger.info(f"Saved chunks ({len(chunks)} items) to {self.chunks_path}")
            
            # 5. Initialize retriever
            logger.info(f"\n[5/5] Initializing hybrid retriever")
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                chunks=chunks,
                reranker_model=self.reranker_model,
                use_reranking=self.use_reranking,
                semantic_weight=self.semantic_weight
            )
            
            logger.info("=" * 80)
            logger.info("✓ Index built successfully!")
            logger.info(f"  - Documents: {len(documents)}")
            logger.info(f"  - Chunks: {len(chunks)}")
            logger.info(f"  - Embedding dimension: {embedding_dim}")
            logger.info(f"  - Index type: flat (IndexFlatIP)")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {e}", exc_info=True)
            return False
    
    def load_index(self) -> bool:
        """
        Load pre-built index from disk.
        
        Returns:
            True if successful
        """
        try:
            logger.info("Loading pre-built index...")

            # Check if metadata and chunks exist
            if not self.metadata_path.exists() or not self.chunks_path.exists():
                logger.error("Index files not found. Run 'build' command first.")
                return False

            # Load embedder
            self.embedder = Embedder(self.embedding_model)

            # Load vector store
            embedding_dim = self.embedder.get_embedding_dimension()

            store_type = self._resolve_vector_store_type()

            if store_type == "pinecone":
                self.vector_store = PineconeVectorStore.load(
                    None,
                    self.metadata_path,
                    dimension=embedding_dim
                )
            else:
                if not self.embeddings_path.exists():
                    logger.error("Embeddings file not found. Run 'build' command first.")
                    return False

                self.vector_store = LocalVectorStore.load(
                    self.embeddings_path,
                    self.metadata_path,
                    dimension=embedding_dim
                )

            # Load chunks
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded chunks ({len(self.chunks)} items) from {self.chunks_path}")
            
            # Load embeddings for reference (optional)
            # embeddings = self.embedder.load_embeddings(self.embeddings_path)
            
            logger.info(f"✓ Loaded index with {self.vector_store.get_total_vectors()} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}", exc_info=True)
            return False
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            explain: Whether to include explanation
            
        Returns:
            Results dictionary with JSON-serializable data
        """
        if not self.retriever:
            raise RuntimeError("Retriever not initialized. Build or load index first.")
        
        start_time = time.time()
        
        try:
            # Perform search
            results = self.retriever.search(
                query,
                top_k=top_k,
                use_hybrid=self.use_hybrid,
                use_reranking=self.use_reranking
            )
            
            # Add explanations if requested
            if explain:
                results = self.retriever.explain_results(query, results)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted = format_search_result(
                    rank=result["rank"],
                    score=result.get("reranked_score", result["score"]),
                    content=result["content"],
                    metadata=result.get("metadata", {})
                )
                
                if explain:
                    formatted["explanation"] = result.get("explanation", {})
                
                formatted_results.append(formatted)
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Format output
            output = format_json_output(
                query=query,
                results=formatted_results,
                execution_time_ms=execution_time_ms,
                top_k=top_k
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            raise


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Semantic Search Engine for Twitter API Postman Documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from documentation
  python semantic_search.py build --clone

  # Search with default settings
  python semantic_search.py search --query "How do I fetch tweets with expansions?"

  # Hybrid search with top 10 results and explanation
  python semantic_search.py search --query "authentication" --top_k 10 --explain

  # Semantic search only (no keyword fusion)
  python semantic_search.py search --query "user lookup" --no-hybrid
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build search index")
    build_parser.add_argument(
        "--clone",
        action="store_true",
        help="Clone documentation from GitHub if not present"
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens"
    )
    build_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Token overlap between chunks"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument(
        "--query",
        required=True,
        help="Search query"
    )
    search_parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to return"
    )
    search_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search (semantic only)"
    )
    search_parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable cross-encoder re-ranking"
    )
    search_parser.add_argument(
        "--explain",
        action="store_true",
        help="Include explanation for results"
    )
    search_parser.add_argument(
        "--output",
        type=str,
        help="Output file (if not provided, prints to stdout)"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create engine
    engine = SemanticSearchEngine()
    
    try:
        if args.command == "build":
            logger.info("Building index...")
            success = engine.build_index(
                clone_docs=args.clone,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            if not success:
                sys.exit(1)
        
        elif args.command == "search":
            # Load existing index
            if not engine.load_index():
                logger.error("Failed to load index. Run 'build' command first.")
                sys.exit(1)
            
            # Prepare retriever
            engine.use_hybrid = not args.no_hybrid
            engine.use_reranking = not args.no_reranking
            
            if engine.vector_store:
                engine.retriever = HybridRetriever(
                    vector_store=engine.vector_store,
                    embedder=engine.embedder,
                    chunks=engine.chunks or [],
                    reranker_model=engine.reranker_model,
                    use_reranking=engine.use_reranking,
                    semantic_weight=engine.semantic_weight
                )
            
            # Perform search
            results = engine.search(
                query=args.query,
                top_k=args.top_k,
                explain=args.explain
            )
            
            # Output results
            output_json = json.dumps(results, indent=2, ensure_ascii=False)
            
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output_json)
                logger.info(f"Results saved to {args.output}")
            else:
                print(output_json)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
