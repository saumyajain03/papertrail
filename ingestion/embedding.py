"""
Embedding generation using sentence-transformers (local, free).
Uses the configured multilingual model from config.py.
"""

from sentence_transformers import SentenceTransformer
import torch

def embed_chunks(chunks: list, model_name: str) -> list:
    """Generate embeddings locally using sentence-transformers."""
    print(f"Loading embedding model: {model_name}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    print("Generating embeddings locally (no API calls)...")
    texts = [chunk["text"] for chunk in chunks]
    
    # Batch encode all chunks at once — much faster than one-by-one
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    
    print(f"Embedded {len(chunks)} chunks successfully.")
    return chunks
