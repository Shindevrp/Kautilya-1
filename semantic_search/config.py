"""
Configuration module for Semantic Search Engine
Loads environment variables and provides application settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
DEVICE = os.getenv("DEVICE", "cpu")

# Vector Index Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "local")
EMBEDDINGS_CACHE_PATH = DATA_DIR / os.getenv("EMBEDDINGS_CACHE_PATH", "embeddings.npy").lstrip("./")
METADATA_CACHE_PATH = DATA_DIR / os.getenv("METADATA_CACHE_PATH", "metadata.pkl").lstrip("./")
CHUNKS_CACHE_PATH = DATA_DIR / os.getenv("CHUNKS_CACHE_PATH", "chunks.pkl").lstrip("./")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "semantic-search")

# Search Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() == "true"
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.4"))

# Query Cache Configuration
QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
QUERY_CACHE_PATH = DATA_DIR / os.getenv("QUERY_CACHE_PATH", "query_cache.db").lstrip("./")
QUERY_CACHE_MAX_SIZE = int(os.getenv("QUERY_CACHE_MAX_SIZE", "1000"))
QUERY_CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "3600"))

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_PATH = LOGS_DIR / os.getenv("LOG_PATH", "semantic_search.log").lstrip("./").split("/")[-1]

# Documentation Configuration
DOCS_PATH = BASE_DIR / "data" / "twitter_api_docs"
DOCS_REPO_URL = os.getenv("DOCS_REPO_URL", "https://github.com/xdevplatform/postman-twitter-api")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Weights for hybrid search
RRF_K = 60  # Reciprocal Rank Fusion constant
