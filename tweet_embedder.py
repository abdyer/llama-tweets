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
    
    def __init__(self, model_name: str = "mxbai-embed-large", collection_name: str = "tweets"):
        """
        Initialize the TweetEmbedder with specified model and collection.
        
        Args:
            model_name: The Ollama embedding model to use
            collection_name: The ChromaDB collection name
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.client = chromadb.Client()
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}'")
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
        
        Args:
            tweets: List of tweet dictionaries to embed
        """
        print(f"Generating embeddings for {len(tweets)} tweets using {self.model_name}...")
        
        for i, tweet in enumerate(tweets):
            try:
                # Extract content for embedding
                tweet_content = tweet['content']
                
                # Generate embedding using Ollama
                response = ollama.embed(model=self.model_name, input=tweet_content)
                embeddings = response["embeddings"][0]  # Extract the first (and only) embedding
                
                # Prepare metadata for storage
                metadata = {
                    "tweet_id": tweet.get('tweet_id', str(i)),
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
                
                # Store in ChromaDB
                self.collection.add(
                    ids=[f"{self.collection_name}_{i}"],
                    embeddings=[embeddings],  # Wrap in list for ChromaDB
                    documents=[tweet_content],
                    metadatas=[metadata]
                )
                
                if (i + 1) % 5 == 0:
                    print(f"Processed {i + 1}/{len(tweets)} tweets")
                    
            except Exception as e:
                print(f"Error processing tweet {i}: {e}")
                continue
        
        print("Embedding process completed!")
    
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
            "model_name": self.model_name
        }
