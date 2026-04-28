from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages: list, chunk_size: int, chunk_overlap: int) -> list:
    """Takes loaded pages and splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        
        if not text.strip():
            continue
            
        page_chunks = text_splitter.split_text(text)
        
        for text_chunk in page_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page["page_number"],
                "text": text_chunk
            })
            chunk_id += 1
            
    print(f"Total chunks created: {len(chunks)}")
    return chunks
