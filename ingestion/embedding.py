"""
Embedding generation using the free Hugging Face Inference API.
This completely removes local memory overhead.
"""

from huggingface_hub import InferenceClient
import config
import time
import streamlit as st

def embed_chunks(chunks: list, model_name: str) -> list:
    """Generate embeddings using Hugging Face API."""
    print(f"Calling Hugging Face Inference API for model: {model_name}")
    client = InferenceClient(token=config.HF_API_KEY)
    
    texts = [chunk["text"] for chunk in chunks]
    
    try:
        # returns numpy array or list
        res = client.feature_extraction(texts, model=model_name)
        embeddings = res.tolist() if hasattr(res, "tolist") else res
    except Exception as e:
        if "503" in str(e) or "loading" in str(e).lower():
            st.warning("Model is waking up on Hugging Face. Waiting 20 seconds...")
            time.sleep(20)
            res = client.feature_extraction(texts, model=model_name)
            embeddings = res.tolist() if hasattr(res, "tolist") else res
        else:
            raise e
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
    
    print(f"Embedded {len(chunks)} chunks via API successfully.")
    return chunks
