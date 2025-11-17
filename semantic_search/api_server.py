#!/usr/bin/env python3
"""
REST API wrapper for Semantic Search Engine
FastAPI server with Swagger documentation
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

from semantic_search import SemanticSearchEngine
from config import API_HOST, API_PORT, API_WORKERS, LOG_LEVEL, LOG_PATH
from utils.helpers import setup_logging


logger = setup_logging(LOG_PATH, LOG_LEVEL)


# Pydantic models for API
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    top_k: int = Field(5, description="Number of results", ge=1, le=20)
    hybrid: bool = Field(True, description="Use hybrid search")
    reranking: bool = Field(True, description="Use cross-encoder re-ranking")
    explain: bool = Field(False, description="Include explanation")


class SearchResult(BaseModel):
    """Individual search result"""
    rank: int
    score: float
    content: str
    metadata: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    top_k: int
    results: List[SearchResult]
    execution_time_ms: float
    timestamp: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    index_loaded: bool
    total_vectors: int = 0


# Create FastAPI app
app = FastAPI(
    title="Semantic Search Engine API",
    description="API for semantic search over Twitter API Postman documentation",
    version="1.0.0"
)

# Global engine instance
engine: Optional[SemanticSearchEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    global engine
    
    logger.info("Initializing search engine...")
    engine = SemanticSearchEngine()
    
    try:
        if engine.load_index():
            logger.info("✓ Index loaded successfully")
            
            # Initialize retriever for API usage
            if engine.vector_store:
                from search.retriever import HybridRetriever
                engine.retriever = HybridRetriever(
                    vector_store=engine.vector_store,
                    embedder=engine.embedder,
                    chunks=engine.chunks or [],
                    reranker_model=engine.reranker_model,
                    use_reranking=engine.use_reranking,
                    semantic_weight=engine.semantic_weight
                )
                logger.info("✓ Retriever initialized successfully")
            else:
                logger.warning("⚠️ Vector store not available")
        else:
            logger.warning("⚠️ Could not load index. Run 'python semantic_search.py build' first")
    except Exception as e:
        logger.error(f"Error loading index: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and index status"""
    if engine is None:
        return HealthResponse(
            status="initializing",
            index_loaded=False
        )
    
    total_vectors = 0
    if engine.vector_store:
        total_vectors = engine.vector_store.get_total_vectors()
    
    return HealthResponse(
        status="healthy",
        index_loaded=engine.retriever is not None,
        total_vectors=total_vectors
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Perform semantic search over documentation
    
    Example:
    ```json
    {
        "query": "How do I fetch tweets with expansions?",
        "top_k": 5,
        "hybrid": true,
        "reranking": true
    }
    ```
    """
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if not engine.retriever:
        raise HTTPException(status_code=503, detail="Index not loaded. Please rebuild index.")
    
    try:
        # Configure search
        engine.use_hybrid = request.hybrid
        engine.use_reranking = request.reranking
        
        # Perform search
        results = engine.search(
            query=request.query,
            top_k=request.top_k,
            explain=request.explain
        )
        
        # Convert to response model
        search_results = [
            SearchResult(
                rank=r["rank"],
                score=r["score"],
                content=r["content"],
                metadata=r["metadata"],
                explanation=r.get("explanation")
            )
            for r in results.get("results", [])
        ]
        
        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            results=search_results,
            execution_time_ms=results.get("execution_time_ms", 0),
            timestamp=results.get("timestamp", 0)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", tags=["Search"])
async def search_get(
    query: str,
    top_k: int = 5,
    hybrid: bool = True,
    reranking: bool = True,
    explain: bool = False
):
    """
    Perform search (GET endpoint)
    
    Parameters:
    - query: Search query
    - top_k: Number of results (1-20)
    - hybrid: Use hybrid search
    - reranking: Use re-ranking
    - explain: Include explanation
    """
    request = SearchRequest(
        query=query,
        top_k=top_k,
        hybrid=hybrid,
        reranking=reranking,
        explain=explain
    )
    return await search(request)


@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Semantic Search Engine",
        "description": "Search Twitter API Postman documentation semantically",
        "version": "1.0.0",
        "endpoints": {
            "search_post": "/search (POST)",
            "search_get": "/search (GET)",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    """Swagger UI documentation"""
    return {"message": "Visit /docs for interactive API documentation"}


def run_server(host: str = API_HOST, port: int = API_PORT, workers: int = 1):
    """
    Run the API server
    
    Args:
        host: Host to bind to
        port: Port to listen on
        workers: Number of worker processes
    """
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers if workers > 1 else 1,
        log_level=LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    host = API_HOST
    port = API_PORT
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    logger.info("=" * 80)
    logger.info("SEMANTIC SEARCH ENGINE - REST API SERVER")
    logger.info("=" * 80)
    logger.info(f"Starting on http://{host}:{port}")
    logger.info(f"API Docs: http://{host}:{port}/docs")
    logger.info("=" * 80)
    
    run_server(host=host, port=port, workers=1)
