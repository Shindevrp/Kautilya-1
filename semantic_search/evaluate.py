#!/usr/bin/env python3
"""
Evaluation suite for semantic search engine
Metrics: MRR, NDCG@5, Recall@10, MAP
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from statistics import mean

import numpy as np

from utils.helpers import (
    calculate_mrr,
    calculate_ndcg,
    calculate_recall_at_k,
    setup_logging
)
from config import LOG_LEVEL, LOG_PATH

logger = setup_logging(LOG_PATH, LOG_LEVEL)


class EvaluationSuite:
    """Evaluation metrics for search results"""
    
    # Golden test set with ground truth
    GOLDEN_QUERIES = [
        {
            "query": "How do I fetch tweets with expansions?",
            "relevant_keywords": ["tweets", "expansions", "lookup", "retrieve"],
            "expected_sections": ["tweets", "lookup", "expansions"],
            "difficulty": "medium"
        },
        {
            "query": "What are the rate limits for user lookup?",
            "relevant_keywords": ["rate", "limit", "user", "lookup"],
            "expected_sections": ["rate limiting", "user", "lookup"],
            "difficulty": "medium"
        },
        {
            "query": "How to authenticate with OAuth 2.0?",
            "relevant_keywords": ["authentication", "OAuth", "bearer", "token"],
            "expected_sections": ["authentication", "OAuth"],
            "difficulty": "easy"
        },
        {
            "query": "Search recent tweets by keyword",
            "relevant_keywords": ["search", "tweets", "recent", "query"],
            "expected_sections": ["search", "tweets", "recent"],
            "difficulty": "medium"
        },
        {
            "query": "Get user followers list",
            "relevant_keywords": ["followers", "user", "list", "lookup"],
            "expected_sections": ["followers", "user"],
            "difficulty": "easy"
        },
        {
            "query": "GET /2/users/:id endpoint parameters",
            "relevant_keywords": ["GET", "/2/users/:id", "parameters", "endpoint"],
            "expected_sections": ["users", "lookup", "parameters"],
            "difficulty": "easy"
        },
        {
            "query": "How to handle pagination with max_results",
            "relevant_keywords": ["pagination", "max_results", "next_token"],
            "expected_sections": ["pagination", "parameters"],
            "difficulty": "hard"
        },
        {
            "query": "Tweet fields and user fields options",
            "relevant_keywords": ["fields", "tweet.fields", "user.fields"],
            "expected_sections": ["fields", "parameters"],
            "difficulty": "hard"
        },
        {
            "query": "Error handling 429 rate limit exceeded",
            "relevant_keywords": ["error", "429", "rate limit", "retry"],
            "expected_sections": ["errors", "rate limiting"],
            "difficulty": "hard"
        },
        {
            "query": "Difference between user lookup and user search",
            "relevant_keywords": ["lookup", "search", "user", "difference"],
            "expected_sections": ["users", "lookup", "search"],
            "difficulty": "hard"
        }
    ]
    
    def __init__(self):
        """Initialize evaluation suite"""
        self.results = []
    
    def evaluate_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate search results against ground truth.
        
        Args:
            query: Original query
            results: List of search results
            ground_truth: Ground truth with relevant keywords
            
        Returns:
            Dictionary of metrics
        """
        # Extract relevant keywords from ground truth
        relevant_keywords = ground_truth.get("relevant_keywords", [])
        
        # Score each result by keyword matching
        scored_results = []
        for result in results:
            content = result.get("content", "").lower()
            relevance_score = sum(
                1 for keyword in relevant_keywords
                if keyword.lower() in content
            ) / len(relevant_keywords) if relevant_keywords else 0
            
            scored_results.append({
                "doc_id": result.get("rank", 0),
                "score": relevance_score,
                "relevant": relevance_score > 0.5  # >50% keyword match = relevant
            })
        
        # Calculate metrics
        metrics = {
            "mrr": self._calculate_mrr(scored_results),
            "ndcg_5": self._calculate_ndcg(scored_results, k=5),
            "recall_10": self._calculate_recall(scored_results, k=10),
            "map": self._calculate_map(scored_results),
            "relevant_count": sum(1 for r in scored_results if r.get("relevant")),
            "total_results": len(results)
        }
        
        return metrics
    
    @staticmethod
    def _calculate_mrr(results: List[Dict[str, Any]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for idx, result in enumerate(results):
            if result.get("relevant", False):
                return 1.0 / (idx + 1)
        return 0.0
    
    @staticmethod
    def _calculate_ndcg(results: List[Dict[str, Any]], k: int = 5) -> float:
        """Calculate NDCG@k"""
        dcg = 0.0
        for idx, result in enumerate(results[:k]):
            if result.get("relevant", False):
                dcg += 1.0 / np.log2(idx + 2)
        
        # Ideal DCG
        ideal_length = min(k, len([r for r in results if r.get("relevant")]))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def _calculate_recall(results: List[Dict[str, Any]], k: int = 10) -> float:
        """Calculate Recall@k"""
        total_relevant = sum(1 for r in results if r.get("relevant"))
        top_k_relevant = sum(1 for r in results[:k] if r.get("relevant"))
        
        return top_k_relevant / total_relevant if total_relevant > 0 else 0.0
    
    @staticmethod
    def _calculate_map(results: List[Dict[str, Any]]) -> float:
        """Calculate Mean Average Precision"""
        precisions = []
        relevant_count = 0
        
        for idx, result in enumerate(results):
            if result.get("relevant", False):
                relevant_count += 1
                precision_at_k = relevant_count / (idx + 1)
                precisions.append(precision_at_k)
        
        return mean(precisions) if precisions else 0.0
    
    def run_full_evaluation(self, searcher) -> Dict[str, Any]:
        """
        Run complete evaluation on golden test set.
        
        Args:
            searcher: Callable that performs search
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("=" * 80)
        logger.info("RUNNING FULL EVALUATION")
        logger.info("=" * 80)
        
        all_metrics = {
            "mrr": [],
            "ndcg_5": [],
            "recall_10": [],
            "map": [],
            "relevant_count": [],
            "queries_evaluated": 0
        }
        
        query_results = []
        
        for test_case in self.GOLDEN_QUERIES:
            query = test_case["query"]
            
            logger.info(f"\nEvaluating: {query}")
            logger.info(f"Difficulty: {test_case['difficulty']}")
            
            try:
                # Perform search
                results = searcher(query, top_k=10)
                
                # Evaluate
                metrics = self.evaluate_results(query, results, test_case)
                
                # Aggregate metrics
                for key in ["mrr", "ndcg_5", "recall_10", "map"]:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
                
                all_metrics["relevant_count"].append(metrics.get("relevant_count", 0))
                all_metrics["queries_evaluated"] += 1
                
                # Log results
                logger.info(
                    f"  MRR: {metrics['mrr']:.4f}, "
                    f"NDCG@5: {metrics['ndcg_5']:.4f}, "
                    f"Recall@10: {metrics['recall_10']:.4f}"
                )
                
                query_results.append({
                    "query": query,
                    "difficulty": test_case["difficulty"],
                    "metrics": metrics,
                    "results_found": len(results)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating query: {e}")
        
        # Calculate averages
        summary = {
            "mean_mrr": mean(all_metrics["mrr"]) if all_metrics["mrr"] else 0,
            "mean_ndcg_5": mean(all_metrics["ndcg_5"]) if all_metrics["ndcg_5"] else 0,
            "mean_recall_10": mean(all_metrics["recall_10"]) if all_metrics["recall_10"] else 0,
            "mean_map": mean(all_metrics["map"]) if all_metrics["map"] else 0,
            "total_queries": len(self.GOLDEN_QUERIES),
            "queries_evaluated": all_metrics["queries_evaluated"],
            "query_results": query_results
        }
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Queries evaluated: {summary['queries_evaluated']}/{summary['total_queries']}")
        logger.info(f"Mean MRR:         {summary['mean_mrr']:.4f}")
        logger.info(f"Mean NDCG@5:      {summary['mean_ndcg_5']:.4f}")
        logger.info(f"Mean Recall@10:   {summary['mean_recall_10']:.4f}")
        logger.info(f"Mean MAP:         {summary['mean_map']:.4f}")
        logger.info("=" * 80)
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save evaluation results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_path}")


def run_evaluation(searcher, output_file: Path = None):
    """
    Run evaluation suite.
    
    Args:
        searcher: Function that performs search (query -> results)
        output_file: Optional file to save results
    """
    evaluator = EvaluationSuite()
    results = evaluator.run_full_evaluation(searcher)
    
    if output_file:
        evaluator.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    # Example usage
    logger.info("Evaluation suite ready. Run with your searcher function.")
