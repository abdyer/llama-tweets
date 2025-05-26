#!/usr/bin/env python3
"""
Simple example demonstrating the Tweet RAG system.
This is a minimal version showing the core functionality.
"""

import ollama
import chromadb

# Sample tweets
sample_tweets = [
    "Just shipped a new AI-powered feature for automatic content categorization. The accuracy is impressive!",
    "Working on a machine learning project that predicts user engagement. Early tests show 85% accuracy.",
    "Diving deep into transformer architectures and their applications in recommendation systems.",
    "Built a RAG application using Ollama. The local embedding models are surprisingly fast.",
    "Optimized our database queries today. Managed to reduce response times by 60% with better indexing.",
]

def run_simple_rag_example():
    """Run a simple RAG example with sample tweets."""
    
    print("🤖 Simple Tweet RAG Example")
    print("=" * 40)
    
    # Initialize ChromaDB
    client = chromadb.Client()
    
    try:
        # Try to delete existing collection if it exists
        client.delete_collection(name="example_tweets")
    except:
        pass
    
    collection = client.create_collection(name="example_tweets")
    
    # Embed and store tweets
    print("📥 Embedding sample tweets...")
    for i, tweet in enumerate(sample_tweets):
        response = ollama.embed(model="mxbai-embed-large", input=tweet)
        embedding = response["embeddings"][0]  # Extract first embedding
        
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[tweet],
            metadatas=[{"tweet_id": i}]
        )
    
    print(f"✅ Embedded {len(sample_tweets)} tweets")
    
    # Query the system
    query = "What performance improvements were made?"
    print(f"\n🔍 Query: {query}")
    
    # Find similar tweets
    query_response = ollama.embed(model="mxbai-embed-large", input=query)
    query_embedding = query_response["embeddings"][0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    relevant_tweets = results["documents"][0]
    print(f"\n📄 Found {len(relevant_tweets)} relevant tweets:")
    for i, tweet in enumerate(relevant_tweets, 1):
        print(f"  {i}. {tweet}")
    
    # Generate response
    context = "\n".join([f"- {tweet}" for tweet in relevant_tweets])
    prompt = f"""Based on these tweets:

{context}

Please answer: {query}"""

    print(f"\n🤖 Generating response...")
    response = ollama.generate(model="llama2", prompt=prompt)
    print(f"\n💬 Response:\n{response['response']}")

if __name__ == "__main__":
    try:
        run_simple_rag_example()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running and models are installed:")
        print("  ollama pull mxbai-embed-large")
        print("  ollama pull llama2")
