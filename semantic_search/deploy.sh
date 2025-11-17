#!/bin/bash

# Semantic Search Engine - Vercel Deployment Script

echo "🚀 Deploying Semantic Search Engine to Vercel"
echo "=============================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Install it first:"
    echo "npm install -g vercel"
    exit 1
fi

# Check if user is logged in to Vercel
if ! vercel whoami &> /dev/null; then
    echo "❌ Not logged in to Vercel. Please run:"
    echo "vercel login"
    exit 1
fi

echo "📦 Step 1: Deploying Python API..."
echo "-----------------------------------"

# Deploy API
API_URL=$(vercel --prod 2>/dev/null | grep -o 'https://[^ ]*\.vercel\.app')

if [ -z "$API_URL" ]; then
    echo "❌ API deployment failed"
    exit 1
fi

echo "✅ API deployed at: $API_URL"

echo ""
echo "📱 Step 2: Deploying Next.js Frontend..."
echo "----------------------------------------"

# Update frontend API URL
sed -i.bak "s|https://your-api-url.vercel.app|$API_URL|g" frontend/vercel.json

# Deploy frontend
cd frontend
FRONTEND_URL=$(vercel --prod 2>/dev/null | grep -o 'https://[^ ]*\.vercel\.app')

if [ -z "$FRONTEND_URL" ]; then
    echo "❌ Frontend deployment failed"
    exit 1
fi

echo "✅ Frontend deployed at: $FRONTEND_URL"

echo ""
echo "🎉 Deployment Complete!"
echo "======================="
echo "🌐 Frontend: $FRONTEND_URL"
echo "🔗 API: $API_URL"
echo ""
echo "📝 Don't forget to set your Pinecone environment variables in Vercel dashboard:"
echo "   - PINECONE_API_KEY"
echo "   - PINECONE_INDEX_NAME"
echo "   - PINECONE_ENVIRONMENT"
echo ""
echo "🔍 Test it out by visiting: $FRONTEND_URL"