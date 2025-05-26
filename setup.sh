#!/bin/bash

echo "🦙 Setting up Tweet RAG Application..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install it from https://ollama.ai/"
    exit 1
fi

echo "✅ Ollama found"

# Pull required models
echo "📥 Pulling embedding model (mxbai-embed-large)..."
ollama pull mxbai-embed-large

echo "📥 Pulling generation model (llama2)..."
ollama pull llama2

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 To get started:"
echo "1. Load your tweets: python rag_app.py --load-tweets"
echo "2. Start interactive chat: python rag_app.py"
echo "3. Or ask a single question: python rag_app.py --query 'What are the main topics in these tweets?'"
