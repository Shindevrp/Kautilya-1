"""
Hybrid search retriever combining semantic and keyword search
Includes BM25 keyword matching and cross-encoder re-ranking
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from utils.helpers import (
    reciprocal_rank_fusion,
    weighted_fusion,
    normalize_scores
)


logger = logging.getLogger("semantic_search")


class HybridRetriever:
    """Hybrid search combining semantic and keyword-based retrieval"""
    
    def __init__(
        self,
        vector_store,
        embedder,
        chunks: List[Dict[str, Any]],
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        use_reranking: bool = True,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: FAISS vector store instance
            embedder: Embedder instance for query embedding
            chunks: List of document chunks
            reranker_model: Cross-encoder model for re-ranking
            use_reranking: Whether to apply cross-encoder re-ranking
            semantic_weight: Weight for semantic search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks
        self.use_reranking = use_reranking
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # Initialize BM25 for keyword search
        logger.info("Initializing BM25 index...")
        self.chunk_texts = [chunk.get("content", "") for chunk in chunks]
        tokenized_chunks = [self._tokenize(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Initialize cross-encoder for re-ranking
        self.reranker = None
        if use_reranking:
            logger.info(f"Loading re-ranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search (semantic only, keyword only, or hybrid).
        
        Args:
            query: User query
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search or just semantic
            use_reranking: Whether to apply cross-encoder re-ranking
            
        Returns:
            List of ranked results with scores and metadata
        """
        if use_reranking is None:
            use_reranking = self.use_reranking
        
        if use_hybrid:
            results = self._hybrid_search(query, top_k * 3)  # Get more for re-ranking
        else:
            results = self._semantic_search(query, top_k)
        
        # Apply re-ranking if enabled
        if use_reranking and self.reranker and len(results) > top_k:
            results = self._rerank_results(query, results)
        
        # Return top-k after re-ranking
        return results[:top_k]
    
    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Perform pure semantic search.
        
        Args:
            query: User query
            k: Number of results
            
        Returns:
            Ranked results
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query, normalize=True)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS
        distances, indices, metadata = self.vector_store.search(query_embedding, k)
        
        # Format results
        results = []
        for rank, (idx, distance, meta) in enumerate(zip(indices, distances, metadata)):
            result = {
                "rank": rank + 1,
                "score": float(distance),
                "content": self.chunks[idx].get("content", ""),
                "metadata": meta,
                "chunk_id": idx,
                "source": "semantic"
            }
            results.append(result)
        
        return results
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: User query
            k: Number of results
            
        Returns:
            Ranked results
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # BM25 search
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                result = {
                    "rank": rank + 1,
                    "score": float(scores[idx]),
                    "content": self.chunks[idx].get("content", ""),
                    "metadata": self.vector_store.get_metadata(idx),
                    "chunk_id": idx,
                    "source": "keyword"
                }
                results.append(result)
        
        return results
    
    def _hybrid_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: User query
            k: Number of results
            
        Returns:
            Fused and ranked results
        """
        logger.debug(f"Performing hybrid search for: {query[:50]}...")
        
        # Get semantic results
        semantic_results = self._semantic_search(query, k)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, k)
        
        # Prepare rankings for RRF
        semantic_rankings = [(r["chunk_id"], r["score"]) for r in semantic_results]
        keyword_rankings = [(r["chunk_id"], r["score"]) for r in keyword_results]
        
        # Apply RRF fusion
        fused_scores = reciprocal_rank_fusion(semantic_rankings, keyword_rankings)
        
        # Create merged results
        all_chunk_ids = set(fused_scores.keys())
        results = []
        
        for chunk_id in all_chunk_ids:
            result = {
                "chunk_id": chunk_id,
                "score": fused_scores[chunk_id],
                "content": self.chunks[chunk_id].get("content", ""),
                "metadata": self.vector_store.get_metadata(chunk_id),
                "source": "hybrid"
            }
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Add ranks
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
        
        return results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply cross-encoder re-ranking to results.
        
        Args:
            query: User query
            results: Initial results to re-rank
            
        Returns:
            Re-ranked results
        """
        if not self.reranker or not results:
            return results
        
        logger.debug(f"Re-ranking {len(results)} results with cross-encoder...")
        
        # Prepare pairs for cross-encoder
        pairs = [
            [query, result["content"]]
            for result in results
        ]
        
        # Get re-ranking scores
        reranking_scores = self.reranker.predict(pairs)
        
        # Update scores and re-rank
        for result, score in zip(results, reranking_scores):
            result["reranked_score"] = float(score)
        
        results.sort(key=lambda x: x["reranked_score"], reverse=True)
        
        # Update ranks
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
        
        return results
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization with lowercasing
        tokens = text.lower().split()
        # Remove common stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
            'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of',
            'on', 'or', 'that', 'the', 'to', 'was', 'will', 'with'
        }
        tokens = [t.strip('.,!?;:') for t in tokens if t.strip() and t.lower() not in stopwords]
        return tokens
    
    def explain_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Provide explanation for why results were ranked.
        
        Args:
            query: User query
            results: Results to explain
            
        Returns:
            Results with explanation added
        """
        explained_results = []
        
        for result in results:
            explanation = {
                "similarity_score": result.get("score", 0.0),
                "reranked_score": result.get("reranked_score", None),
                "source": result.get("source", "unknown"),
                "chunk_metadata": result.get("metadata", {}),
            }
            
            result["explanation"] = explanation
            explained_results.append(result)
        
        return explained_results
