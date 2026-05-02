"""
Embedding generation using fastembed (lightweight ONNX runtime).
Uses the configured multilingual model from config.py.
"""

from fastembed import TextEmbedding

def embed_chunks(chunks: list, model_name: str) -> list:
    """Generate embeddings locally using fastembed."""
    print(f"Loading lightweight embedding model: {model_name}")
    # fastembed downloads and caches the model automatically
    model = TextEmbedding(model_name=model_name, threads=1)
    
    print("Generating embeddings locally (no API calls, low memory)...")
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings generator and convert to list
    embeddings_generator = model.embed(texts)
    embeddings = list(embeddings_generator)
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    
    print(f"Embedded {len(chunks)} chunks successfully.")
    return chunks
