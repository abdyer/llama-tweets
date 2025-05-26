# Tweet RAG Application

A **Retrieval Augmented Generation (RAG)** application that analyzes markdown tweet archives using Ollama models. The system embeds tweets using local language models, stores them in a vector database, and answers questions based on the most relevant tweet content.

## 🚀 Features

- **Markdown Tweet Archive Support**: Processes tweet archives exported in markdown format
- **Rich Metadata Extraction**: Extracts timestamps, URLs, and tweet IDs from markdown archives
- **Persistent Vector Storage**: ChromaDB with persistent storage to avoid re-processing tweets
- **Incremental Updates**: Automatically skips already embedded tweets for efficient processing
- **Local LLM Integration**: Uses Ollama for both embeddings and text generation
- **Vector Search**: ChromaDB for efficient similarity search
- **Interactive Chat**: Command-line interface for querying tweet history
- **Batch Processing**: Load and embed multiple tweet files efficiently
- **Flexible Models**: Configurable embedding and generation models

## 📋 Prerequisites

1. **Install Ollama** from https://ollama.ai/
2. **Pull required models**:
   ```bash
   ollama pull mxbai-embed-large  # Embedding model
   ollama pull llama2             # Generation model
   ```

## ⚙️ Setup

1. **Clone/navigate to the project directory**
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your tweet archives**:
   - Place your markdown tweet archive files (`.md`) in the `data/` directory
   - The system will automatically detect and process all `.md` files in the directory

## 🎯 Usage

### Basic Usage
Load tweets and start interactive chat:
```bash
python rag_app.py --load-tweets
```

### Single Query
Process one query without interactive mode:
```bash
python rag_app.py --load-tweets --query "What technologies are mentioned?"
```

### Advanced Options
```bash
python rag_app.py \
  --tweets-file data/your-tweet-archive.md \
  --embedding-model mxbai-embed-large \
  --generation-model llama2 \
  --load-tweets
```

### Testing
Run the comprehensive test suite:
```bash
python test_rag.py
```

The test script will:
- Automatically find markdown files in the `data/` directory
- Test the complete RAG workflow with sample queries
- Verify metadata extraction and error handling
- Display collection statistics and sample results

## 📁 Project Structure

```
llama-tweets/
├── rag_app.py              # Main RAG application
├── tweet_embedder.py       # Embedding and vector storage
├── test_rag.py            # Test suite
├── requirements.txt        # Python dependencies
├── data/                   # Place your markdown tweet archives here
└── README.md              # This file
```

## 🔧 Configuration

### Models
- **Embedding Model**: `mxbai-embed-large` (default)
- **Generation Model**: `llama2` (default)

### Data Format
Tweet archives should be in markdown format with the following structure:
```markdown
> Tweet content here

![Image](timestamp-info) [Timestamp](https://twitter.com/username/status/tweetid)

----

> Another tweet content

![Image](timestamp-info) [Timestamp](https://twitter.com/username/status/tweetid)

----
```

Example markdown format:
```markdown
> I have a headache.  I never have headaches.

![Image](screenshot) [Tue Aug 05 14:19:12 +0000 2008](https://twitter.com/username/status/878282969)

----
```

## 🧪 Example Queries

- "What technologies are mentioned in the tweets?"
- "Tell me about AI and machine learning projects"
- "What performance improvements were made?"
- "What programming activities are described?"

## 🛠️ Technical Details

### Architecture
1. **Tweet Loading**: Parse tweets from markdown archive files with metadata extraction
2. **Metadata Processing**: Extract timestamps, URLs, and tweet IDs using regex parsing
3. **Embedding Generation**: Use Ollama's `mxbai-embed-large` model
4. **Vector Storage**: Store embeddings in ChromaDB with rich metadata
5. **Query Processing**: Embed user query and find similar tweets
6. **Response Generation**: Use Ollama's `llama2` model with contextual tweet history

### Dependencies
- `ollama`: Local LLM integration
- `chromadb`: Vector database
- `python-dotenv`: Environment configuration

## 🚧 Future Enhancements

- [ ] Web interface using Streamlit or FastAPI
- [ ] Support for multiple document types (PDFs, web pages)
- [ ] Advanced filtering and search options
- [ ] Export/import functionality for vector databases
- [ ] Integration with social media APIs
- [ ] Conversation memory and context persistence
