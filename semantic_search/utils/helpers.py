"""
Utility helper functions for Semantic Search Engine
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

import numpy as np


def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_path: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("semantic_search")
    logger.setLevel(getattr(logging, level.upper()))
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def format_search_result(
    rank: int,
    score: float,
    content: str,
    metadata: Dict[str, Any],
    execution_time_ms: float = None
) -> Dict[str, Any]:
    """
    Format a single search result.
    
    Args:
        rank: Result rank (1-indexed)
        score: Similarity score (0-1)
        content: Document content/text
        metadata: Document metadata
        execution_time_ms: Query execution time
        
    Returns:
        Formatted result dictionary
    """
    return {
        "rank": rank,
        "score": round(float(score), 4),
        "content": content[:500] + "..." if len(content) > 500 else content,
        "content_full": content,
        "metadata": metadata
    }


def format_json_output(
    query: str,
    results: List[Dict[str, Any]],
    execution_time_ms: float,
    top_k: int = None
) -> Dict[str, Any]:
    """
    Format final JSON output for CLI.
    
    Args:
        query: User query
        results: List of result dictionaries
        execution_time_ms: Query execution time in milliseconds
        top_k: Number of results requested
        
    Returns:
        Formatted JSON-ready dictionary
    """
    return {
        "query": query,
        "top_k": top_k or len(results),
        "results": results,
        "execution_time_ms": round(execution_time_ms, 2),
        "timestamp": time.time()
    }


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Array of scores
        method: Normalization method ("minmax" or "zscore")
        
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    if method == "minmax":
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    elif method == "zscore":
        mean = scores.mean()
        std = scores.std()
        if std == 0:
            return np.ones_like(scores)
        normalized = (scores - mean) / std
        # Clip to [-3, 3] and scale to [0, 1]
        return np.clip(normalized, -3, 3) / 6 + 0.5
    
    return scores


def calculate_mrr(results: List[Dict[str, Any]], ground_truth_indices: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        results: Ranked list of results
        ground_truth_indices: Indices of relevant documents
        
    Returns:
        MRR score (0-1)
    """
    if not ground_truth_indices or not results:
        return 0.0
    
    for idx, result in enumerate(results):
        if result.get("doc_id") in ground_truth_indices:
            return 1.0 / (idx + 1)
    
    return 0.0


def calculate_ndcg(results: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        results: Ranked list of results (must have 'relevant' field)
        k: Number of top results to consider
        
    Returns:
        NDCG@k score (0-1)
    """
    if not results:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for idx, result in enumerate(results[:k]):
        if result.get("relevant", False):
            dcg += 1.0 / np.log2(idx + 2)
    
    # Calculate ideal DCG (all relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(results))))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_recall_at_k(
    results: List[Dict[str, Any]], 
    ground_truth_indices: List[int],
    k: int = 10
) -> float:
    """
    Calculate Recall@k.
    
    Args:
        results: Ranked list of results
        ground_truth_indices: Indices of relevant documents
        k: Number of top results to consider
        
    Returns:
        Recall@k score (0-1)
    """
    if not ground_truth_indices:
        return 0.0
    
    top_k_ids = [r.get("doc_id") for r in results[:k]]
    relevant_found = len(set(top_k_ids) & set(ground_truth_indices))
    
    return relevant_found / len(ground_truth_indices)


def reciprocal_rank_fusion(
    semantic_results: List[Tuple[int, float]],
    keyword_results: List[Tuple[int, float]],
    k: int = 60
) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple rankings.
    
    Args:
        semantic_results: List of (doc_id, rank) from semantic search
        keyword_results: List of (doc_id, rank) from keyword search
        k: RRF constant (default 60)
        
    Returns:
        Dictionary of {doc_id: fused_score}
    """
    fused_scores = {}
    
    # Add semantic search results
    for rank, (doc_id, score) in enumerate(semantic_results, 1):
        fused_scores[doc_id] = 1.0 / (k + rank)
    
    # Add keyword search results
    for rank, (doc_id, score) in enumerate(keyword_results, 1):
        if doc_id in fused_scores:
            fused_scores[doc_id] += 1.0 / (k + rank)
        else:
            fused_scores[doc_id] = 1.0 / (k + rank)
    
    return fused_scores


def weighted_fusion(
    semantic_scores: Dict[int, float],
    keyword_scores: Dict[int, float],
    semantic_weight: float = 0.6
) -> Dict[int, float]:
    """
    Weighted fusion of semantic and keyword search scores.
    
    Args:
        semantic_scores: Dictionary of {doc_id: score} from semantic search
        keyword_scores: Dictionary of {doc_id: score} from keyword search
        semantic_weight: Weight for semantic scores (0-1)
        
    Returns:
        Dictionary of {doc_id: fused_score}
    """
    keyword_weight = 1.0 - semantic_weight
    fused_scores = {}
    
    all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    for doc_id in all_doc_ids:
        sem_score = semantic_scores.get(doc_id, 0.0)
        kw_score = keyword_scores.get(doc_id, 0.0)
        
        fused_scores[doc_id] = (
            sem_score * semantic_weight + 
            kw_score * keyword_weight
        )
    
    return fused_scores


def chunks_to_list(text: str, chunk_size: int) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters.
    
    Args:
        text: Text to chunk
        chunk_size: Approximate chunk size
        
    Returns:
        List of text chunks
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (1 token ≈ 4 characters).
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    return len(text) // 4
