import os
from api_server import app
from mangum import Mangum

# Vercel expects a handler function
handler = Mangum(app, lifespan="off")

# Set environment variables for Vercel
if "VERCEL" in os.environ:
    # Vercel environment variables
    os.environ.setdefault("PINECONE_API_KEY", os.environ.get("PINECONE_API_KEY", ""))
    os.environ.setdefault("PINECONE_INDEX_NAME", os.environ.get("PINECONE_INDEX_NAME", "semantic-search-kautilya"))
    os.environ.setdefault("PINECONE_ENVIRONMENT", os.environ.get("PINECONE_ENVIRONMENT", "us-east-1"))