#!/usr/bin/env python3
"""
Quick start guide and test script for Semantic Search Engine
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command"""
    print(f"\n{'='*70}")
    print(f"📍 {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    """Interactive setup and testing guide"""
    
    project_root = Path(__file__).parent
    
    print("\n" + "="*70)
    print("🚀 SEMANTIC SEARCH ENGINE - QUICK START GUIDE")
    print("="*70)
    
    print("""
This tool will help you get started with the Semantic Search Engine.

Features:
  ✅ Semantic search using BAAI/bge-large-en-v1.5
  ✅ Hybrid search (semantic + BM25 keyword)
  ✅ Cross-encoder re-ranking
  ✅ Pinecone vector indexing
  ✅ REST API with FastAPI
  ✅ Docker containerization
  ✅ Comprehensive evaluation metrics

Next steps:

1. INSTALL DEPENDENCIES
   Run: bash setup.sh

2. BUILD INDEX
   Run: python semantic_search.py build --clone
   
3. RUN SEARCHES
   Basic:  python semantic_search.py search --query "your query"
   Advanced: python semantic_search.py search --query "query" --top_k 10 --explain

4. RUN DEMO
   Run: python demo.py

5. START API SERVER
   Run: python api_server.py
   Then visit: http://localhost:8000/docs

6. EVALUATE
   Run: python evaluate.py

7. DOCKER DEPLOYMENT
   Build:  docker build -t semantic-search .
   Run:    docker run -p 8000:8000 semantic-search
   Or:     docker-compose up

================================================================================
""")
    
    # Check current status
    print("\n📊 CURRENT STATUS")
    print("-"*70)
    
    # Check docs
    docs_path = project_root / "data" / "twitter_api_docs"
    if docs_path.exists():
        print("✓ Documentation found")
        md_count = len(list(docs_path.glob("**/*.md")))
        json_count = len(list(docs_path.glob("**/*.json")))
        print(f"  - Markdown files: {md_count}")
        print(f"  - JSON files: {json_count}")
    else:
        print("✗ Documentation not found")
        print("  Run: git clone https://github.com/xdevplatform/postman-twitter-api data/twitter_api_docs")
    
    # Check index
    metadata_path = project_root / "data" / "data" / "metadata.pkl"
    chunks_path = project_root / "data" / "data" / "chunks.pkl"
    if metadata_path.exists() and chunks_path.exists():
        print("✓ Pinecone index built")
    else:
        print("✗ Pinecone index not built")
        print("  Run: python semantic_search.py build --clone")
    
    # Check logs
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        print("✓ Logs directory ready")
    
    print("\n")


if __name__ == "__main__":
    main()
