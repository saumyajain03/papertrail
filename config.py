import os
from dotenv import load_dotenv

load_dotenv()

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding Configuration (fastembed — lightweight ONNX)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM Configuration (Groq — free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"

# Vector Store Configuration (ChromaDB)
VECTORSTORE_PATH = "./vectorstore"
COLLECTION_NAME = "pdf_chunks"
