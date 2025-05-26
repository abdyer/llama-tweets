---
mode: 'agent'
---

We are going to build a retrieval augmented generation (RAG) application using Ollama and embedding models. This will involve generating vector embeddings from tweet history, storing them in a database, retrieving relevant documents based on a query, and then generating a response using those documents.

Follow the instructions below to complete the task.

## Ollama Embedding Models

Ollama supports embedding models, making it possible to build retrieval augmented generation (RAG) applications that combine text prompts with existing documents or other data.

### What are embedding models?

Embedding models are models that are trained specifically to generate vector embeddings: long arrays of numbers that represent semantic meaning for a given sequence of text.

The resulting vector embedding arrays can then be stored in a database, which will compare them as a way to search for data that is similar in meaning.

### Example embedding models

| Model | Parameter Size |
|-------|----------------|
| mxbai-embed-large | 334M |
| nomic-embed-text | 137M |
| all-minilm | 23M |

### Usage

To generate vector embeddings, first pull a model:

```bash
ollama pull mxbai-embed-large
```

Next, use the REST API, Python or JavaScript libraries to generate vector embeddings from the model:

**REST API**

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "mxbai-embed-large",
  "input": "Llamas are members of the camelid family"
}'
```

**Python library**

```python
ollama.embed(
  model='mxbai-embed-large',
  input='Llamas are members of the camelid family',
)
```

**Javascript library**

```javascript
ollama.embed({
    model: 'mxbai-embed-large',
    input: 'Llamas are members of the camelid family',
})
```

Ollama also integrates with popular tooling to support embeddings workflows such as LangChain and LlamaIndex.

## Example

This example walks through building a retrieval augmented generation (RAG) application using Ollama and embedding models.

### Step 1: Generate embeddings

```bash
pip install ollama chromadb
```

Create a file named `example.py` with the contents:

```python
import ollama
import chromadb

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embed(model="mxbai-embed-large", input=d)
  embeddings = response["embeddings"]
  collection.add(
    ids=[str(i)],
    embeddings=embeddings,
    documents=[d]
  )
```

### Step 2: Retrieve

Next, add the code to retrieve the most relevant document given an example prompt:

```python
# an example input
input = "What animals are llamas related to?"

# generate an embedding for the input and retrieve the most relevant doc
response = ollama.embed(
  model="mxbai-embed-large",
  input=input
)
results = collection.query(
  query_embeddings=[response["embeddings"]],
  n_results=1
)
data = results['documents'][0][0]
```

### Step 3: Generate

Lastly, use the prompt and the document retrieved in the previous step to generate an answer!

```python
# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama2",
  prompt=f"Using this data: {data}. Respond to this prompt: {input}"
)

print(output['response'])
```

Then, run the code:

```bash
python example.py
```

Llama 2 will answer the prompt "What animals are llamas related to?" using the data:

> Llamas are members of the camelid family, which means they are closely related to two other animals: vicuñas and camels. All three species belong to the same evolutionary lineage and share many similarities in terms of their physical characteristics, behavior, and genetic makeup. Specifically, llamas are most closely related to vicuñas, with which they share a common ancestor that lived around 20-30 million years ago. Both llamas and vicuñas are members of the family Camelidae, while camels belong to a different family (Dromedary).

## Coming soon

More features are coming to support workflows that involve embeddings:

- **Batch embeddings**: processing multiple input data prompts simultaneously
- **OpenAI API Compatibility**: support for the `/v1/embeddings` OpenAI-compatible endpoint
- **More embedding model architectures**: support for ColBERT, RoBERTa, and other embedding model architectures