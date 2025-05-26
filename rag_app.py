import ollama
from tweet_embedder import TweetEmbedder
import argparse
import os

class TweetRAG:
    """
    A Retrieval Augmented Generation application for tweets using Ollama.
    """
    
    def __init__(self, embedding_model: str = "mxbai-embed-large", 
                 generation_model: str = "llama2"):
        """
        Initialize the RAG application.
        
        Args:
            embedding_model: Model used for generating embeddings
            generation_model: Model used for text generation
        """
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.embedder = TweetEmbedder(model_name=embedding_model)
        
    def load_and_embed_tweets(self, tweets_file: str = None, tweets_dir: str = None, 
                             file_pattern: str = "*.md") -> None:
        """
        Load tweets from markdown file(s) and generate embeddings.
        
        Args:
            tweets_file: Path to a single markdown tweets file
            tweets_dir: Path to directory containing markdown tweet files
            file_pattern: Glob pattern for files in directory (default: *.md)
        """
        if tweets_dir:
            # Load from directory
            tweets = self.embedder.load_tweets_from_directory(tweets_dir, file_pattern)
        elif tweets_file:
            # Load from single file
            tweets = self.embedder.load_tweets_from_file(tweets_file)
        else:
            raise ValueError("Either tweets_file or tweets_dir must be provided")
        
        if not tweets:
            print("No tweets loaded!")
            return
        
        self.embedder.embed_tweets(tweets)
        
        # Display collection info
        info = self.embedder.get_collection_info()
        print(f"\nCollection Info:")
        print(f"- Name: {info['collection_name']}")
        print(f"- Documents: {info['document_count']}")
        print(f"- Embedding Model: {info['model_name']}")
        
        # Show sample of loaded tweets
        print(f"\nSample tweets:")
        for i, tweet in enumerate(tweets[:3]):
            content = tweet['content']
            print(f"  {i+1}. {content[:80]}{'...' if len(content) > 80 else ''}")
            if tweet.get('timestamp'):
                print(f"     📅 {tweet['timestamp']}")
        
        if len(tweets) > 3:
            print(f"  ... and {len(tweets) - 3} more tweets")
    
    def generate_response(self, query: str, n_context_tweets: int = 3) -> str:
        """
        Generate a response to a query using relevant tweets as context.
        
        Args:
            query: The user's question or prompt
            n_context_tweets: Number of similar tweets to use as context
            
        Returns:
            Generated response string
        """
        # Find similar tweets
        search_results = self.embedder.search_similar_tweets(query, n_context_tweets)
        relevant_tweets = search_results["documents"]
        
        print(f"\nFound {len(relevant_tweets)} relevant tweets:")
        for i, tweet in enumerate(relevant_tweets, 1):
            print(f"{i}. {tweet[:100]}...")
        
        # Combine tweets into context
        context = "\n".join([f"- {tweet}" for tweet in relevant_tweets])
        
        # Create prompt for generation
        prompt = f"""Based on the following tweets:

{context}

Please respond to this question: {query}

Use the information from the tweets to provide a relevant and helpful response."""
        
        print(f"\nGenerating response using {self.generation_model}...")
        
        # Generate response
        try:
            response = ollama.generate(
                model=self.generation_model,
                prompt=prompt
            )
            return response["response"]
        except Exception as e:
            return f"Error generating response: {e}"
    
    def interactive_chat(self):
        """
        Start an interactive chat session.
        """
        print("\n🤖 Tweet RAG Chat - Ask questions about the tweet history!")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            try:
                response = self.generate_response(query)
                print(f"\n🤖 Response:\n{response}\n")
                print("-" * 50)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Tweet RAG Application")
    parser.add_argument("--tweets-file", 
                       help="Path to a markdown tweets file (.md)")
    parser.add_argument("--tweets-dir", default="data",
                       help="Path to directory containing markdown tweet files (default: data)")
    parser.add_argument("--file-pattern", default="*.md",
                       help="File pattern for directory loading (default: *.md)")
    parser.add_argument("--embedding-model", default="mxbai-embed-large",
                       help="Ollama embedding model to use")
    parser.add_argument("--generation-model", default="llama2",
                       help="Ollama generation model to use")
    parser.add_argument("--load-tweets", action="store_true",
                       help="Load and embed tweets from file(s)")
    parser.add_argument("--query", type=str,
                       help="Single query to process (non-interactive)")
    
    args = parser.parse_args()
    
    # Initialize RAG application
    rag = TweetRAG(
        embedding_model=args.embedding_model,
        generation_model=args.generation_model
    )
    
    # Load tweets if requested
    if args.load_tweets:
        if args.tweets_file:
            # Load from specific file
            if not os.path.exists(args.tweets_file):
                print(f"❌ Tweets file not found: {args.tweets_file}")
                return
            if not args.tweets_file.lower().endswith('.md'):
                print(f"❌ Only markdown (.md) files are supported: {args.tweets_file}")
                return
            print(f"📥 Loading tweets from {args.tweets_file}...")
            rag.load_and_embed_tweets(tweets_file=args.tweets_file)
        else:
            # Load from directory
            if not os.path.exists(args.tweets_dir):
                print(f"❌ Tweets directory not found: {args.tweets_dir}")
                return
            print(f"📥 Loading tweets from directory: {args.tweets_dir} (pattern: {args.file_pattern})...")
            rag.load_and_embed_tweets(tweets_dir=args.tweets_dir, file_pattern=args.file_pattern)
        
        print("✅ Tweets loaded and embedded successfully!")
    
    # Process single query or start interactive chat
    if args.query:
        response = rag.generate_response(args.query)
        print(f"\n🤖 Response:\n{response}")
    else:
        rag.interactive_chat()

if __name__ == "__main__":
    main()
