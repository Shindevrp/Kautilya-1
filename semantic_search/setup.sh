#!/bin/bash
# Installation and setup helper script

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📦 Semantic Search Engine - Setup Helper"
echo "Project directory: $PROJECT_DIR"
echo "=================================================="

# Check Python version
echo ""
echo "✓ Checking Python..."
python3 --version

# Install requirements
echo ""
echo "✓ Installing dependencies..."
pip install -q sentence-transformers faiss-cpu rank-bm25 numpy pandas tqdm python-dotenv PyYAML requests transformers torch scikit-learn pytest fastapi uvicorn

echo ""
echo "✓ Verifying installations..."
python3 -c "import sentence_transformers, faiss, numpy; print('✓ All core dependencies installed!')"

# Check/create directories
echo ""
echo "✓ Creating data directories..."
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Build index: python semantic_search.py build --clone"
echo "  2. Search:     python semantic_search.py search --query 'your query'"
echo "  3. Demo:       python demo.py"
echo "  4. Evaluate:   python evaluate.py"
echo ""
