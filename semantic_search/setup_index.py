#!/usr/bin/env python3
"""
Setup script to build the semantic search index
Run: python setup_index.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str = ""):
    """Run a shell command and report results."""
    if description:
        print(f"\n{'='*80}")
        print(f"► {description}")
        print(f"{'='*80}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    """Build semantic search index."""
    base_dir = Path(__file__).parent.absolute()
    
    print(f"\n{'='*80}")
    print("SEMANTIC SEARCH ENGINE - INDEX BUILDER")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    
    # Verify venv exists
    venv_python = base_dir / "venv" / "bin" / "python3"
    if not venv_python.exists():
        print(f"\n✗ Virtual environment not found at {venv_python}")
        print("Run: python3 -m venv venv")
        return False
    
    # Build index
    cmd = f"{venv_python} {base_dir}/semantic_search.py build --clone --chunk-size 512 --chunk-overlap 64"
    
    if not run_command(cmd, "Building Semantic Search Index"):
        return False
    
    print(f"\n{'='*80}")
    print("✓ Index built successfully!")
    print(f"{'='*80}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
