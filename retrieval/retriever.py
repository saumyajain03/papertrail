"""
Retrieval module using ChromaDB as the vector store.
Chunks are stored with their fastembed embeddings and page number metadata.
Queries are embedded locally and searched via ChromaDB's query interface.
"""

import chromadb
from fastembed import TextEmbedding
import config
from langdetect import detect

# Load the embedding model once — shared across indexing and querying
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=config.EMBEDDING_MODEL, threads=1)
    return _model


def build_vectorstore(chunks: list) -> chromadb.Collection:
    """
    Takes embedded chunks, stores them in ChromaDB, and returns the collection.
    Persists data to the local vectorstore/ folder.
    """
    client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)

    # Drop and recreate collection so it's always fresh for a new PDF
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # use cosine distance
    )

    print(f"Adding {len(chunks)} chunks to ChromaDB collection '{config.COLLECTION_NAME}'...")

    ids         = [str(chunk["chunk_id"]) for chunk in chunks]
    documents   = [chunk["text"] for chunk in chunks]
    embeddings  = [chunk["embedding"] for chunk in chunks]
    metadatas   = [{"page_number": chunk["page_number"]} for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"ChromaDB collection built with {collection.count()} entries.")
    return collection


def detect_language(query: str) -> str:
    """Detect the language of the query."""
    try:
        return detect(query)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"


def search_query(query: str, collection: chromadb.Collection, top_k: int = 3) -> list:
    """
    Embed the query locally via SBERT and search ChromaDB for the top_k closest chunks.
    Returns a list of dicts with text, page_number, and distance score.
    """
    language = detect_language(query)
    print(f"Detected language: {language}")

    model = _get_model()
    print(f"\nEmbedding query: '{query}'")
    query_embedding_generator = model.embed([query])
    query_embedding = list(query_embedding_generator)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Reformat ChromaDB results into the same structure the rest of the pipeline expects
    formatted = []
    for i in range(len(results["ids"][0])):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score: 1 - distance (so 1 = identical)
        distance = results["distances"][0][i]
        score = 1 - distance

        formatted.append({
            "chunk_id": results["ids"][0][i],
            "page_number": results["metadatas"][0][i]["page_number"],
            "text": results["documents"][0][i],
            "score": score
        })

    return formatted
