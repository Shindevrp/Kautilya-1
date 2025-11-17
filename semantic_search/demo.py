#!/usr/bin/env python3
"""
Demo script showcasing semantic search capabilities
Compares baseline semantic, hybrid search, and re-ranked results
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from config import LOG_LEVEL, LOG_PATH
from utils.helpers import setup_logging


logger = setup_logging(LOG_PATH, LOG_LEVEL)


class SearchDemo:
    """Demo showcasing search capabilities"""
    
    # Diverse test queries covering different scenarios
    TEST_QUERIES = [
        {
            "query": "GET /2/users/:id/followers",
            "type": "endpoint_exact",
            "description": "Exact API endpoint match"
        },
        {
            "query": "How do I retrieve a user's followers?",
            "type": "semantic_similarity",
            "description": "Semantic variation of endpoint query"
        },
        {
            "query": "authentication stuff",
            "type": "vague_query",
            "description": "Vague query requiring query enhancement"
        },
        {
            "query": "example OAuth implementation",
            "type": "code_search",
            "description": "Looking for code examples"
        },
        {
            "query": "difference between user lookup and user search",
            "type": "comparison",
            "description": "Requires reasoning across multiple documents"
        },
        {
            "query": "rate limiting 429 error",
            "type": "error_handling",
            "description": "Error debugging query"
        },
        {
            "query": "pagination with max_results and next_token",
            "type": "parameter_query",
            "description": "Parameter and feature interaction"
        },
        {
            "query": "What fields can I include for tweets?",
            "type": "field_discovery",
            "description": "Field enumeration query"
        }
    ]
    
    def __init__(self, searcher=None):
        """
        Initialize demo.
        
        Args:
            searcher: Function that performs search (optional)
        """
        self.searcher = searcher
    
    def run_demo(self, mode: str = "full"):
        """
        Run demonstration.
        
        Args:
            mode: "full" (all features) or "quick" (basic)
        """
        logger.info("=" * 90)
        logger.info("🚀 SEMANTIC SEARCH ENGINE - FEATURE DEMONSTRATION")
        logger.info("=" * 90)
        
        if not self.searcher:
            logger.warning(
                "No searcher provided. Run demo after building index:\n"
                "  python semantic_search.py build --clone\n"
                "  python demo.py"
            )
            return
        
        # Run queries
        for i, test_case in enumerate(self.TEST_QUERIES[:5 if mode == "quick" else len(self.TEST_QUERIES)], 1):
            self._demo_query(test_case, i)
            logger.info("-" * 90)
        
        logger.info("=" * 90)
        logger.info("✓ Demo completed!")
        logger.info("=" * 90)
    
    def _demo_query(self, test_case: Dict[str, Any], query_num: int) -> None:
        """
        Demonstrate search for a single query.
        
        Args:
            test_case: Query test case
            query_num: Query number for display
        """
        query = test_case["query"]
        query_type = test_case.get("type", "unknown")
        description = test_case.get("description", "")
        
        logger.info(f"\n[Query {query_num}] {query_type.upper()}")
        logger.info(f"Description: {description}")
        logger.info(f"Query: \"{query}\"")
        logger.info("-" * 90)
        
        try:
            # 1. Baseline Semantic Search
            logger.info("\n1️⃣  SEMANTIC SEARCH (baseline)")
            start = time.time()
            results_semantic = self.searcher(query, top_k=5, mode="semantic")
            time_semantic = time.time() - start
            
            self._print_results(results_semantic, time_semantic)
            
            # 2. Hybrid Search
            logger.info("\n2️⃣  HYBRID SEARCH (semantic + BM25 keyword)")
            start = time.time()
            results_hybrid = self.searcher(query, top_k=5, mode="hybrid")
            time_hybrid = time.time() - start
            
            self._print_results(results_hybrid, time_hybrid)
            
            # 3. Hybrid + Re-ranking
            logger.info("\n3️⃣  HYBRID + RE-RANKING (cross-encoder boost)")
            start = time.time()
            results_reranked = self.searcher(query, top_k=5, mode="hybrid_reranked")
            time_reranked = time.time() - start
            
            self._print_results(results_reranked, time_reranked)
            
            # 4. Comparison
            logger.info("\n📊 COMPARISON")
            self._compare_results(results_semantic, results_hybrid, results_reranked)
            
        except Exception as e:
            logger.error(f"Error during demo: {e}", exc_info=True)
    
    @staticmethod
    def _print_results(results: List[Dict[str, Any]], exec_time: float) -> None:
        """Print search results nicely"""
        logger.info(f"Execution time: {exec_time*1000:.1f}ms\n")
        
        if not results:
            logger.info("  (No results found)")
            return
        
        for result in results:
            rank = result.get("rank", 0)
            score = result.get("score", 0)
            content = result.get("content", "")[:150]
            source = result.get("metadata", {}).get("source", "unknown")
            
            logger.info(f"  [{rank}] Score: {score:.4f} | Source: {source}")
            logger.info(f"      → {content}...")
    
    @staticmethod
    def _compare_results(
        baseline: List[Dict[str, Any]],
        hybrid: List[Dict[str, Any]],
        reranked: List[Dict[str, Any]]
    ) -> None:
        """Compare results from different methods"""
        
        # Extract top-1 from each
        top1_baseline = baseline[0] if baseline else None
        top1_hybrid = hybrid[0] if hybrid else None
        top1_reranked = reranked[0] if reranked else None
        
        logger.info("  Top-1 Ranking Comparison:")
        logger.info(f"    Semantic:         {(top1_baseline['score'] if top1_baseline else 'N/A')}")
        logger.info(f"    Hybrid:           {(top1_hybrid['score'] if top1_hybrid else 'N/A')}")
        logger.info(f"    Hybrid+Rerank:    {(top1_reranked['score'] if top1_reranked else 'N/A')}")
        
        # Check if top-1 changed
        if baseline and hybrid:
            if baseline[0].get("metadata") != hybrid[0].get("metadata"):
                logger.info("  ✓ Hybrid reordered results (keyword matching added value)")
        
        if hybrid and reranked:
            if hybrid[0].get("metadata") != reranked[0].get("metadata"):
                logger.info("  ✓ Re-ranking boosted relevance (cross-encoder improved ranking)")


def create_searcher_wrapper(engine):
    """
    Create a searcher function for demo.
    
    Args:
        engine: SemanticSearchEngine instance
        
    Returns:
        Searcher function
    """
    def search(query: str, top_k: int = 5, mode: str = "semantic"):
        """Search with different modes"""
        if mode == "semantic":
            result = engine.search(query, top_k, explain=False)
            return result.get("results", [])
        elif mode == "hybrid":
            engine.use_hybrid = True
            engine.use_reranking = False
            result = engine.search(query, top_k, explain=False)
            return result.get("results", [])
        elif mode == "hybrid_reranked":
            engine.use_hybrid = True
            engine.use_reranking = True
            result = engine.search(query, top_k, explain=False)
            return result.get("results", [])
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    return search


def main():
    """Run demo"""
    from semantic_search import SemanticSearchEngine
    from search.retriever import HybridRetriever
    
    # Initialize engine
    logger.info("Initializing search engine...")
    engine = SemanticSearchEngine()
    
    # Load index
    if not engine.load_index():
        logger.error("Failed to load index. Run 'python semantic_search.py build --clone' first")
        return
    
    # Initialize retriever (required for search)
    if engine.vector_store:
        engine.retriever = HybridRetriever(
            vector_store=engine.vector_store,
            embedder=engine.embedder,
            chunks=engine.chunks or [],
            reranker_model=engine.reranker_model,
            use_reranking=engine.use_reranking,
            semantic_weight=engine.semantic_weight
        )
    
    # Create searcher wrapper
    searcher = create_searcher_wrapper(engine)
    
    # Run demo
    demo = SearchDemo(searcher=searcher)
    demo.run_demo(mode="full")


if __name__ == "__main__":
    main()
