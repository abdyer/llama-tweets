#!/usr/bin/env python3
"""
Test script for the Tweet RAG application with markdown tweet archives.
"""

from rag_app import TweetRAG
import os
import glob

def test_rag_system():
    """Test the complete RAG system with markdown tweet archives."""
    
    print("🧪 Testing Tweet RAG System (Markdown Archives)")
    print("=" * 50)
    
    # Initialize RAG system
    rag = TweetRAG()
    
    # Check if any markdown files exist in data directory
    data_dir = "data"
    markdown_files = glob.glob(os.path.join(data_dir, "*.md"))
    
    if not markdown_files:
        print(f"❌ No markdown tweet archives found in {data_dir}/")
        print("💡 Please add your markdown tweet archive files (.md) to the data/ directory")
        return
    
    print(f"📁 Found {len(markdown_files)} markdown file(s): {[os.path.basename(f) for f in markdown_files]}")
    
    # Load and embed tweets from directory
    print("\n📥 Loading and embedding tweets from markdown archives...")
    rag.load_and_embed_tweets(tweets_dir=data_dir)
    
    # Test queries relevant to tweet archives
    test_queries = [
        "What topics does the author tweet about?",
        "What music or entertainment does the author enjoy?",
        "What programming languages or technologies are mentioned?",
        "What personal interests or activities are described?"
    ]
    
    print("\n🔍 Testing queries...")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🤔 Query {i}: {query}")
        try:
            response = rag.generate_response(query, n_context_tweets=2)
            print(f"🤖 Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 30)
    
    print("\n✅ RAG system test completed!")
    
    # Display collection info
    info = rag.embedder.get_collection_info()
    print(f"\n📊 Collection Summary:")
    print(f"   - Total documents: {info['document_count']}")
    print(f"   - Embedding model: {info['model_name']}")

def test_individual_components():
    """Test individual components of the system."""
    
    print("\n🔧 Testing Individual Components")
    print("=" * 50)
    
    from tweet_embedder import TweetEmbedder
    
    # Test embedder with markdown files
    embedder = TweetEmbedder(collection_name="test_components")
    
    data_dir = "data"
    markdown_files = glob.glob(os.path.join(data_dir, "*.md"))
    
    if markdown_files:
        test_file = markdown_files[0]
        print(f"\n🧪 Testing embedder with: {os.path.basename(test_file)}")
        
        try:
            tweets = embedder.load_tweets_from_file(test_file)
            print(f"✅ Loaded {len(tweets)} tweets with metadata")
            
            # Test metadata extraction
            if tweets:
                sample_tweet = tweets[0]
                print(f"✅ Sample metadata:")
                print(f"   - Tweet ID: {sample_tweet.get('tweet_id', 'N/A')}")
                print(f"   - Timestamp: {sample_tweet.get('timestamp', 'N/A')}")
                print(f"   - Content preview: {sample_tweet['content'][:50]}...")
                
        except Exception as e:
            print(f"❌ Component test failed: {e}")
    
    # Test error handling
    print(f"\n🧪 Testing error handling...")
    try:
        embedder.load_tweets_from_file("data/tweets.txt")
        print("❌ Should have rejected .txt file")
    except ValueError as e:
        print(f"✅ Correctly rejected .txt file: {e}")
    
    print("\n✅ Component tests completed!")

if __name__ == "__main__":
    test_rag_system()
    test_individual_components()
