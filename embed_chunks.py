import boto3
import json
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


# Load documents again
pdf_files = [
    "data/Elie Wiesel - Night FULL TEXT.pdf",
    "data/How To Find Your Inner Happiness.pdf"
]
all_documents = []
for pdf in pdf_files:
    loader = PyMuPDFLoader(pdf)
    all_documents.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_documents)
texts = [chunk.page_content for chunk in chunks]

# Titan Embedder via Bedrock
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_titan_embedding(text):
    body = json.dumps({
        "inputText": text
    })
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        body=body
    )
    result = json.loads(response["body"].read())
    return np.array(result['embedding'], dtype='float32')

# Embed all texts
embeddings = [get_titan_embedding(t) for t in texts]

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and texts
faiss.write_index(index, "faiss_index.index")

with open("texts.json", "w", encoding='utf-8') as f:
    json.dump(texts, f, ensure_ascii=False)

print("✅ FAISS index created and saved.")
