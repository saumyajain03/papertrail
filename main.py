import sys
import os
import config
from ingestion import loader, chunking, embedding
from retrieval import retriever
from generation import llm

def run_ingestion_pipeline(pdf_path: str):
    print("--- Starting Pipeline ---")
    
    if not config.GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set in environment or .env file.")
        sys.exit(1)
        
    # Step 1: Load
    pages = loader.load_pdf(pdf_path)
    
    # Step 2: Chunk
    chunks = chunking.chunk_text(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    # Step 3: Embed
    embedded_chunks = embedding.embed_chunks(chunks, config.EMBEDDING_MODEL)
    
    if embedded_chunks:
        first = embedded_chunks[0]
        print("\n--- Ingestion Successful ---")
        print(f"Sample - Chunk {first['chunk_id']} (Page {first['page_number']})")
        print(f"Text Preview: {first['text'][:50]}...")
        if "embedding" in first and first["embedding"]:
            print(f"Embedding Dimensions: {len(first['embedding'])}")
            print(f"First 5 Dimensions: {first['embedding'][:5]}")
            
    return embedded_chunks

if __name__ == "__main__":
    test_pdf = "data/Attention is all you need.pdf"
    if not os.path.exists(test_pdf):
        print(f"Test PDF not found at {test_pdf}. Please add one.")
        sys.exit(1)

    embedded_chunks = run_ingestion_pipeline(test_pdf)
    if not embedded_chunks:
        print("Ingestion failed. Exiting.")
        sys.exit(1)

    print("\n--- Building ChromaDB Vector Store ---")
    collection = retriever.build_vectorstore(embedded_chunks)

    # Conversation history — stores alternating user/assistant dicts
    conversation_history = []
    last_chunks = []  # Cache last retrieved chunks for follow-up fallback

    REFUSAL_THRESHOLD = 0.25  # If best score < this, reuse last chunks or refuse

    test_queries = [
        "What is the Transformer model and how does it work?",
        "Can you elaborate on that?",           # follow-up — vague, needs last chunks
        "What is the capital of France?",       # out-of-scope — should refuse
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print(f"History length: {len(conversation_history)} messages")
        print("=" * 60)

        # Step 1: Retrieve relevant chunks
        top_chunks = retriever.search_query(query, collection, top_k=3)
        best_score = top_chunks[0]["score"] if top_chunks else 0

        print("\nRetrieved Chunks:")
        for i, chunk in enumerate(top_chunks):
            print(f"  [{i+1}] Page {chunk['page_number']} | Score: {chunk['score']:.4f}")

        # If score is too low AND we have prior chunks, fall back to them (follow-up case)
        if best_score < REFUSAL_THRESHOLD and last_chunks:
            print(f"\n[Low score: {best_score:.4f}] Reusing previous chunks for context.")
            top_chunks = last_chunks
        elif best_score >= REFUSAL_THRESHOLD:
            last_chunks = top_chunks  # Cache good chunks for potential follow-ups

        # Step 2: Generate answer with conversation history
        print("\nGenerating answer...")
        answer = llm.generate_answer(query, top_chunks, conversation_history)

        print(f"\nANSWER:\n{answer}")
        print("-" * 60)

        # Step 3: Append this turn to conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": answer})



