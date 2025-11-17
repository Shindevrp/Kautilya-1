"""Vector store implementations used by the semantic search engine.

This module provides both the Pinecone-backed vector store as well as a
lightweight local numpy-based fallback that can be used when a Pinecone API
key is not available. Both stores expose the same interface so the rest of the
codebase can stay agnostic of the underlying backend.
"""

import logging
import pickle
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

try:  # Pinecone is optional when running with the local backend
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
    _PINECONE_AVAILABLE = True
except ImportError:  # pragma: no cover - happens in lightweight installs
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore
    _PINECONE_AVAILABLE = False


logger = logging.getLogger("semantic_search")


class PineconeVectorStore:
    """Pinecone-based vector index with metadata support"""

    def __init__(self, dimension: int, index_name: str = None, api_key: str = None):
        """
        Initialize Pinecone vector store.

        Args:
            dimension: Embedding dimension
            index_name: Name of the Pinecone index
            api_key: Pinecone API key
        """
        self.dimension = dimension
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.api_key = api_key or PINECONE_API_KEY

        if not self.api_key:
            raise ValueError("Pinecone API key not provided")
        if not _PINECONE_AVAILABLE:
            raise ImportError("pinecone package is not installed")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Check if index exists, create if not
        if self.index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(self.index_name)
        self.metadata = []  # Keep local metadata for compatibility

        logger.info(f"Connected to Pinecone index: {self.index_name}, dimension={dimension}")

    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add embeddings to index with metadata.

        Args:
            embeddings: Numpy array of embeddings (N, dimension)
            metadata: List of metadata dictionaries for each embedding
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) "
                f"must match metadata count ({len(metadata)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) "
                f"must match index dimension ({self.dimension})"
            )

        # Prepare vectors for Pinecone
        vectors = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            vector_id = f"vec_{i}"
            # Add chunk_id to metadata for retrieval
            meta_with_id = {**meta, "chunk_id": i}
            vectors.append((vector_id, embedding.tolist(), meta_with_id))

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)

        self.metadata.extend(metadata)

        logger.info(f"Added {len(metadata)} vectors to Pinecone index")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Search for nearest neighbors.

        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices, metadata_list)
        """
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension ({query_embedding.shape[1]}) "
                f"must match index dimension ({self.dimension})"
            )

        # Search Pinecone
        results = self.index.query(
            vector=query_embedding.flatten().tolist(),
            top_k=k,
            include_metadata=True
        )

        # Extract results
        distances = []
        indices = []
        metadata_list = []

        for match in results['matches']:
            distances.append(match['score'])
            # Extract chunk_id from metadata
            chunk_id = match['metadata'].pop('chunk_id', 0)
            indices.append(int(chunk_id))
            metadata_list.append(match['metadata'])

        return np.array(distances), np.array(indices, dtype=int), metadata_list

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]]:
        """
        Batch search for multiple queries.

        Args:
            query_embeddings: Array of query embeddings (N, dimension)
            k: Number of nearest neighbors for each query

        Returns:
            List of (distances, indices, metadata) tuples
        """
        results = []
        for query_emb in query_embeddings:
            dist, idx, meta = self.search(query_emb.reshape(1, -1), k)
            results.append((dist, idx, meta))
        return results

    def save(self, index_path: Optional[Path], metadata_path: Path) -> None:
        """
        Save metadata locally (Pinecone handles the index).

        Args:
            index_path: Not used for Pinecone
            metadata_path: Path to save metadata
        """
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata ({len(self.metadata)} items) to {metadata_path}")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path, dimension: int = None, index_name: str = None, api_key: str = None) -> "PineconeVectorStore":
        """
        Load from Pinecone index.

        Args:
            index_path: Not used for Pinecone
            metadata_path: Path to metadata file
            dimension: Embedding dimension
            index_name: Pinecone index name
            api_key: Pinecone API key

        Returns:
            Loaded PineconeVectorStore instance
        """
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"Loaded metadata ({len(metadata)} items) from {metadata_path}")

        # Create instance
        store = cls(dimension=dimension, index_name=index_name, api_key=api_key)
        store.metadata = metadata

        return store

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific vector index."""
        if idx < len(self.metadata):
            return self.metadata[idx]
        return {}

    def get_total_vectors(self) -> int:
        """Get total number of vectors in index."""
        stats = self.index.describe_index_stats()
        return stats['total_vector_count']

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class LocalVectorStore:
    """Lightweight in-memory vector store backed by numpy arrays."""

    def __init__(
        self,
        dimension: int,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self.dimension = dimension
        self.embeddings = embeddings
        self.metadata = metadata or []

    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) must match metadata count ({len(metadata)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) must match index dimension ({self.dimension})"
            )

        # Store embeddings as float32 for numerical stability
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata

        logger.info(f"Added {len(metadata)} vectors to local index")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if self.embeddings is None:
            raise ValueError("No embeddings available in the local vector store")

        if query_embedding.ndim == 2:
            query_vec = query_embedding[0]
        else:
            query_vec = query_embedding

        if query_vec.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension ({query_vec.shape[0]}) must match index dimension ({self.dimension})"
            )

        scores = np.dot(self.embeddings, query_vec.astype(np.float32))
        k = min(k, scores.shape[0])
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        distances = scores[top_indices]
        metadata_list = [self.get_metadata(idx) for idx in top_indices]

        return distances.astype(np.float32), top_indices.astype(int), metadata_list

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]]:
        results = []
        for query_emb in query_embeddings:
            dist, idx, meta = self.search(query_emb.reshape(1, -1), k)
            results.append((dist, idx, meta))
        return results

    def save(self, index_path: Optional[Path], metadata_path: Path) -> None:
        if index_path is None:
            raise ValueError("Local vector store requires an index path to save embeddings")

        if self.embeddings is None:
            raise ValueError("No embeddings to save")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(index_path), self.embeddings)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        logger.info(
            f"Saved local index embeddings to {index_path} and metadata ({len(self.metadata)} items) to {metadata_path}"
        )

    @classmethod
    def load(
        cls,
        index_path: Path,
        metadata_path: Path,
        dimension: Optional[int] = None
    ) -> "LocalVectorStore":
        embeddings = np.load(str(index_path))

        if dimension is None:
            dimension = embeddings.shape[1]

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        logger.info(
            f"Loaded local index with {embeddings.shape[0]} vectors from {index_path} and metadata from {metadata_path}"
        )

        return cls(dimension=dimension, embeddings=embeddings.astype(np.float32), metadata=metadata)

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        if idx < len(self.metadata):
            return self.metadata[idx]
        return {}

    def get_total_vectors(self) -> int:
        return 0 if self.embeddings is None else int(self.embeddings.shape[0])

    def get_dimension(self) -> int:
        return self.dimension