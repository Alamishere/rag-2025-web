# RAG 2025 Web

This is a simple Retrieval-Augmented Generation (RAG) project built with:
-  **FAISS** for semantic search
-  **Amazon Titan Embeddings** for vector representation
-  **Claude 3 Sonnet (via Bedrock)** for answering questions
-  PDF data sources with context-aware question answering

---

## How it Works

1. **Embed PDF chunks** with Titan Embeddings
2. **Index** them with FAISS
3. **User asks a question** → Titan embeds it → Top 3 chunks retrieved
4. **Claude 3** (via Bedrock) generates an answer using those chunks

---

##  Files

| File | Purpose |
|------|---------|
| `load_pdf.py` | Loads and chunks PDF files |
| `embed_chunks.py` | Converts chunks to Titan embeddings and builds FAISS index |
| `faiss_index.index` | Vector database of all embedded chunks |
| `texts.json` | Original text chunks used for context |
| `app.py` | Streamlit web interface for RAG chat with conversation history and real-time document search |
| `ask.py` | Query engine using Claude 3 with retrieved context |
| `data/` | Contains PDF source files |

---

##  How to Run

### 1. Set up virtual environment
 
```bash
python -m venv rag-env
.\rag-env\Scripts\activate  # on Windows

## Run the Assistant

```bash
python ask.py

Example: Ask a question: What is the book Night about?

Answer:Based on the context provided, Night is a memoir by Elie Wiesel about...

## Notes
You must have access to Amazon Bedrock and be whitelisted for Claude 3.

The modelId used for Claude 3.5 Sonnet is:

anthropic.claude-3-sonnet-20240229-v1:0
Amazon Titan embedding model used:

amazon.titan-embed-text-v1
If you get an AccessDeniedException, make sure your IAM permissions and Bedrock access are configured.

