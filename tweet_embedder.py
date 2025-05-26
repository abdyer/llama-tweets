import ollama
import chromadb
import os
import re
import glob
from typing import List, Dict, Any, Optional, Tuple

class TweetEmbedder:
    """
    A class to handle tweet embedding and storage using Ollama and ChromaDB.
    """
    
    def __init__(self, model_name: str = "mxbai-embed-large", collection_name: str = "tweets", 
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the TweetEmbedder with specified model and collection.
        
        Args:
            model_name: The Ollama embedding model to use
            collection_name: The ChromaDB collection name
            persist_directory: Directory to store the persistent ChromaDB database
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            count = self.collection.count()
            print(f"Loaded existing collection '{collection_name}' with {count} documents")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection '{collection_name}'")
    
    def load_tweets_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load tweets from a markdown tweet archive file.
        
        Args:
            file_path: Path to the markdown tweets file
            
        Returns:
            List of tweet dictionaries with content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tweet file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext != '.md':
            raise ValueError(f"Only markdown (.md) files are supported. Got: {file_ext}")
        
        return self._load_tweets_from_markdown(file_path)
    
    def _load_tweets_from_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Load tweets from markdown format with metadata."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tweets = []
        
        # Split by ---- separators
        sections = content.split('----')
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Extract tweet content (blockquote)
            tweet_content = self._extract_tweet_content(section)
            if not tweet_content:
                continue
            
            # Extract metadata
            metadata = self._extract_tweet_metadata(section)
            
            tweets.append({
                'content': tweet_content,
                'tweet_id': metadata.get('tweet_id', f"md_{i}"),
                'timestamp': metadata.get('timestamp'),
                'url': metadata.get('url'),
                'source_file': file_path,
                'format': 'markdown'
            })
        
        print(f"Loaded {len(tweets)} tweets from markdown file: {file_path}")
        return tweets
    
    def _extract_tweet_content(self, section: str) -> Optional[str]:
        """Extract tweet content from a markdown section."""
        # Look for blockquote content (> text)
        lines = section.split('\n')
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                # Remove the > and any leading whitespace
                content = line[1:].strip()
                if content:
                    content_lines.append(content)
        
        return ' '.join(content_lines) if content_lines else None
    
    def _extract_tweet_metadata(self, section: str) -> Dict[str, str]:
        """Extract metadata from a markdown section."""
        metadata = {}
        
        # Extract timestamp and URL from the image tag line
        # Pattern: [Tue Aug 05 14:19:12 +0000 2008](https://twitter.com/dammitandy/status/878282969)
        url_pattern = r'\[([^\]]+)\]\((https://twitter\.com/[^)]+/status/(\d+))\)'
        match = re.search(url_pattern, section)
        
        if match:
            metadata['timestamp'] = match.group(1)
            metadata['url'] = match.group(2)
            metadata['tweet_id'] = match.group(3)
        
        return metadata
    
    def load_tweets_from_directory(self, directory_path: str, file_pattern: str = "*.md") -> List[Dict[str, Any]]:
        """
        Load tweets from multiple files in a directory.
        
        Args:
            directory_path: Path to directory containing tweet files
            file_pattern: Glob pattern for files to load (default: *.md)
            
        Returns:
            List of tweet dictionaries from all matching files
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pattern = os.path.join(directory_path, file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            print(f"No files found matching pattern: {pattern}")
            return []
        
        all_tweets = []
        for file_path in sorted(files):
            try:
                tweets = self.load_tweets_from_file(file_path)
                all_tweets.extend(tweets)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Total tweets loaded from {len(files)} files: {len(all_tweets)}")
        return all_tweets
    
    def embed_tweets(self, tweets: List[Dict[str, Any]]) -> None:
        """
        Generate embeddings for tweets and store them in ChromaDB.
        Only processes tweets that haven't been embedded yet.
        
        Args:
            tweets: List of tweet dictionaries to embed
        """
        print(f"Checking {len(tweets)} tweets for embedding using {self.model_name}...")
        
        # Check for already embedded tweets
        embedded_tweet_ids = self._get_embedded_tweet_ids()
        print(f"Found {len(embedded_tweet_ids)} already embedded tweets")
        
        # Filter out already embedded tweets
        new_tweets = []
        skipped_count = 0
        
        for tweet in tweets:
            tweet_id = tweet.get('tweet_id', f"unknown_{len(new_tweets)}")
            if tweet_id in embedded_tweet_ids:
                skipped_count += 1
            else:
                new_tweets.append(tweet)
        
        print(f"Skipping {skipped_count} already embedded tweets")
        print(f"Processing {len(new_tweets)} new tweets...")
        
        if not new_tweets:
            print("No new tweets to embed!")
            return
        
        processed_count = 0
        for tweet in new_tweets:
            try:
                # Extract content for embedding
                tweet_content = tweet['content']
                tweet_id = tweet.get('tweet_id', f"unknown_{processed_count}")
                
                # Generate embedding using Ollama
                response = ollama.embed(model=self.model_name, input=tweet_content)
                embeddings = response["embeddings"][0]  # Extract the first (and only) embedding
                
                # Prepare metadata for storage
                metadata = {
                    "tweet_id": tweet_id,
                    "length": len(tweet_content),
                    "source_file": tweet.get('source_file', 'unknown'),
                    "format": tweet.get('format', 'markdown')
                }
                
                # Add timestamp if available
                if tweet.get('timestamp'):
                    metadata['timestamp'] = tweet['timestamp']
                
                # Add URL if available
                if tweet.get('url'):
                    metadata['url'] = tweet['url']
                
                # Use tweet_id as the document ID for consistent deduplication
                doc_id = f"tweet_{tweet_id}"
                
                # Store in ChromaDB
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embeddings],  # Wrap in list for ChromaDB
                    documents=[tweet_content],
                    metadatas=[metadata]
                )
                
                processed_count += 1
                if processed_count % 5 == 0:
                    print(f"Processed {processed_count}/{len(new_tweets)} new tweets")
                    
            except Exception as e:
                print(f"Error processing tweet {tweet_id}: {e}")
                continue
        
        total_count = self.collection.count()
        print(f"Embedding process completed! Added {processed_count} new tweets.")
        print(f"Total tweets in collection: {total_count}")
    
    def search_similar_tweets(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search for tweets similar to the given query.
        
        Args:
            query: The search query
            n_results: Number of similar tweets to return
            
        Returns:
            Dictionary containing the search results
        """
        print(f"Searching for tweets similar to: '{query}'")
        
        # Generate embedding for the query
        response = ollama.embed(model=self.model_name, input=query)
        query_embedding = response["embeddings"][0]  # Extract the first embedding
        
        # Search for similar tweets
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            "query": query,
            "documents": results["documents"][0],
            "distances": results["distances"][0] if "distances" in results else None,
            "metadatas": results["metadatas"][0] if "metadatas" in results else None
        }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "model_name": self.model_name,
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        Use with caution!
        """
        try:
            # Get all document IDs
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Cleared {len(results['ids'])} documents from collection '{self.collection_name}'")
            else:
                print(f"Collection '{self.collection_name}' is already empty")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def _is_tweet_already_embedded(self, tweet_id: str) -> bool:
        """Check if a tweet is already embedded in the collection."""
        try:
            results = self.collection.get(
                where={"tweet_id": tweet_id}
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def _get_embedded_tweet_ids(self) -> set:
        """Get all tweet IDs that are already embedded."""
        try:
            results = self.collection.get()
            if 'metadatas' in results and results['metadatas']:
                tweet_ids = set()
                for metadata in results['metadatas']:
                    if metadata and metadata.get('tweet_id'):
                        tweet_ids.add(metadata['tweet_id'])
                return tweet_ids
            return set()
        except Exception as e:
            print(f"Warning: Could not retrieve embedded tweet IDs: {e}")
            return set()
