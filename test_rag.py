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
        embedder.load_tweets_from_file("nonexistent.txt")
        print("❌ Should have rejected .txt file")
    except ValueError as e:
        print(f"✅ Correctly rejected .txt file: {e}")
    except FileNotFoundError as e:
        print(f"✅ Correctly handled missing file: {e}")
    
    print("\n✅ Component tests completed!")

def test_persistent_storage():
    """Test the persistent storage functionality."""
    
    print("\n💾 Testing Persistent Storage")
    print("=" * 50)
    
    from tweet_embedder import TweetEmbedder
    
    # Test with a dedicated test collection
    embedder = TweetEmbedder(collection_name="test_persistence", persist_directory="./test_chroma_db")
    
    initial_info = embedder.get_collection_info()
    print(f"📊 Initial collection state: {initial_info['document_count']} documents")
    
    data_dir = "data"
    markdown_files = glob.glob(os.path.join(data_dir, "*.md"))
    
    if markdown_files:
        test_file = markdown_files[0]
        print(f"\n📥 Loading tweets from: {os.path.basename(test_file)}")
        
        try:
            # First load
            tweets = embedder.load_tweets_from_file(test_file)
            print(f"✅ Loaded {len(tweets)} tweets")
            
            if tweets:
                # Embed a subset for testing
                test_tweets = tweets[:3] if len(tweets) > 3 else tweets
                print(f"🔄 Embedding {len(test_tweets)} test tweets...")
                embedder.embed_tweets(test_tweets)
                
                after_first_info = embedder.get_collection_info()
                print(f"📊 After first embedding: {after_first_info['document_count']} documents")
                
                # Try to embed the same tweets again (should skip)
                print(f"🔄 Attempting to embed same tweets again...")
                embedder.embed_tweets(test_tweets)
                
                after_second_info = embedder.get_collection_info()
                print(f"📊 After second embedding: {after_second_info['document_count']} documents")
                
                if after_first_info['document_count'] == after_second_info['document_count']:
                    print("✅ Persistent storage correctly avoided duplicates!")
                else:
                    print("❌ Persistent storage allowed duplicates")
                
                # Test searching
                if after_second_info['document_count'] > 0:
                    print(f"\n🔍 Testing search functionality...")
                    search_results = embedder.search_similar_tweets("test query", n_results=1)
                    if search_results['documents']:
                        print(f"✅ Search returned {len(search_results['documents'])} results")
                    else:
                        print("❌ Search returned no results")
                
                # Clean up test collection
                print(f"\n🧹 Cleaning up test collection...")
                embedder.clear_collection()
                final_info = embedder.get_collection_info()
                print(f"📊 After cleanup: {final_info['document_count']} documents")
                
        except Exception as e:
            print(f"❌ Persistent storage test failed: {e}")
    else:
        print("❌ No markdown files available for testing")
    
    print("\n✅ Persistent storage test completed!")

if __name__ == "__main__":
    test_rag_system()
    test_individual_components()
    test_persistent_storage()
