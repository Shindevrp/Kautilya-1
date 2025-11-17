"""
Embedding generation module
Uses sentence-transformers for efficient batch embedding
"""

import logging
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


logger = logging.getLogger("semantic_search")


class Embedder:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu"):
        """
        Initialize embedder with pre-trained model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            normalize: Whether to L2 normalize
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embedding
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (N, embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Embed chunks extracted from documents.
        
        Args:
            chunks: List of chunk dictionaries with 'content' field
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize
            
        Returns:
            Dictionary mapping chunk index to embedding
        """
        # Extract content from chunks
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(
            texts,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=True
        )
        
        # Create mapping
        chunk_embeddings = {i: embeddings[i] for i in range(len(embeddings))}
        
        logger.info(f"Created embeddings for {len(chunk_embeddings)} chunks")
        
        return chunk_embeddings
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: Path
    ) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Numpy array of embeddings
            path: Path to save to
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), embeddings)
        logger.info(f"Saved embeddings ({embeddings.shape}) to {path}")
    
    def load_embeddings(self, path: Path) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            path: Path to embeddings file
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = np.load(str(path))
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
