#!/usr/bin/env python3
"""
Verification and test script
Checks all components are working correctly
"""

import sys
from pathlib import Path


def check_files():
    """Check all required files exist"""
    print("\n📁 Checking project files...")
    
    project_root = Path(__file__).parent
    required_files = [
        "semantic_search.py",
        "api_server.py",
        "demo.py",
        "evaluate.py",
        "config.py",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
        "IMPLEMENTATION_GUIDE.md",
        "requirements.txt",
        ".env"
    ]
    
    required_dirs = [
        "ingestion",
        "embeddings",
        "index",
        "search",
        "utils",
        "data",
        "logs"
    ]
    
    all_good = True
    
    for f in required_files:
        path = project_root / f
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {f}")
        if not path.exists():
            all_good = False
    
    for d in required_dirs:
        path = project_root / d
        status = "✓" if path.is_dir() else "✗"
        print(f"  {status} {d}/")
        if not path.is_dir():
            all_good = False
    
    return all_good


def check_documentation():
    """Check if documentation exists"""
    print("\n📚 Checking documentation...")
    
    project_root = Path(__file__).parent
    docs_path = project_root / "data" / "twitter_api_docs"
    
    if docs_path.exists():
        md_files = list(docs_path.glob("**/*.md"))
        json_files = list(docs_path.glob("**/*.json"))
        print(f"  ✓ Documentation found")
        print(f"    - Markdown files: {len(md_files)}")
        print(f"    - JSON files: {len(json_files)}")
        return True
    else:
        print(f"  ✗ Documentation not found at {docs_path}")
        print(f"    Run: git clone https://github.com/xdevplatform/postman-twitter-api data/twitter_api_docs")
        return False


def check_dependencies():
    """Check if key dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    dependencies = {
        "numpy": "Data processing",
        "pandas": "Data manipulation",
        "tqdm": "Progress bars",
    }
    
    # These might not be installed yet - optional check
    optional = {
        "sentence_transformers": "Embeddings",
        "pinecone": "Vector indexing",
        "rank_bm25": "Keyword search",
        "fastapi": "REST API",
    }
    
    all_good = True
    
    for pkg, desc in dependencies.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} ({desc})")
        except ImportError:
            print(f"  ✗ {pkg} ({desc})")
            all_good = False
    
    print("\n  Optional dependencies:")
    for pkg, desc in optional.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} ({desc})")
        except ImportError:
            print(f"  ⚠ {pkg} ({desc}) - Install with: pip install -r requirements.txt")

    return all_good


def check_structure():
    """Check code structure"""
    print("\n🏗️  Checking code structure...")
    
    project_root = Path(__file__).parent
    
    modules = {
        "ingestion/loader.py": "DocumentationLoader",
        "ingestion/chunker.py": "SemanticChunker",
        "embeddings/embedder.py": "Embedder",
        "index/vector_store.py": "PineconeVectorStore",
        "search/retriever.py": "HybridRetriever",
        "utils/helpers.py": "setup_logging",
    }
    
    all_good = True
    
    for file_path, class_name in modules.items():
        full_path = project_root / file_path
        if full_path.exists():
            content = full_path.read_text()
            if class_name in content:
                print(f"  ✓ {file_path}")
            else:
                print(f"  ⚠ {file_path} (missing {class_name})")
                all_good = False
        else:
            print(f"  ✗ {file_path}")
            all_good = False
    
    return all_good


def main():
    """Run all checks"""
    print("=" * 70)
    print("🔍 SEMANTIC SEARCH ENGINE - VERIFICATION")
    print("=" * 70)
    
    checks = [
        ("Project Files", check_files),
        ("Code Structure", check_structure),
        ("Documentation", check_documentation),
        ("Dependencies", check_dependencies),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    all_good = all(results.values())
    
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    if all_good:
        print("\n✅ Everything looks good!")
        print("\nNext steps:")
        print("1. Install dependencies: bash setup.sh")
        print("2. Build index: python semantic_search.py build --clone")
        print("3. Test search: python semantic_search.py search --query 'your query'")
    else:
        print("\n⚠️  Some checks failed. See above for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
