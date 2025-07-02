from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# List of PDF file paths
pdf_files = [
    "data/Elie Wiesel - Night FULL TEXT.pdf",
    "data/How To Find Your Inner Happiness.pdf"
]

print("Starting PDF load...")

# ---- define the accumulator first ----
all_documents = []

# Load each PDF
for pdf in pdf_files:
    print(f"Loading: {pdf}")
    loader = PyMuPDFLoader(pdf)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from: {pdf}")
    all_documents.extend(docs)

print(f"Total combined pages: {len(all_documents)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_documents)

print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)
