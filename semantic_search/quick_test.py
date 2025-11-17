#!/usr/bin/env python3
"""
Quick demo script to test the semantic search pipeline
Uses simpler models and processes a subset of docs
"""

import os
import sys
import json
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🚀 Semantic Search Engine - Quick Demo")
    print("=" * 80)
    
    # Check if docs exist
    docs_path = project_root / "data" / "twitter_api_docs"
    if not docs_path.exists():
        print(f"❌ Documentation not found at {docs_path}")
        print("Please run: git clone https://github.com/xdevplatform/postman-twitter-api data/twitter_api_docs")
        return False
    
    print(f"✓ Found documentation at {docs_path}")
    
    # List available files
    md_files = list(docs_path.glob("**/*.md"))
    json_files = list(docs_path.glob("**/*.json"))
    
    print(f"✓ Found {len(md_files)} Markdown files")
    print(f"✓ Found {len(json_files)} JSON files")
    
    # Show sample files
    print("\nSample files found:")
    for f in list(md_files + json_files)[:5]:
        rel_path = f.relative_to(docs_path)
        print(f"  - {rel_path}")
    
    print("\n" + "=" * 80)
    print("✓ Documentation ready for processing!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Build index: python semantic_search.py build --clone")
    print("  3. Search: python semantic_search.py search --query 'your query here'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
