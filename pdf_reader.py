"""
Step 2: PDF Reader and Chunker
Reads a PDF file using PyMuPDF (fitz), extracts text from each page,
and splits the text into chunks using manual string slicing.
"""

import sys
import fitz  # PyMuPDF


def read_and_chunk_pdf(file_path: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Read a PDF and split text into chunks with overlap."""
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)

    print(f"PDF: {file_path}")
    print(f"Total pages: {len(doc)}")
    
    chunks = []
    chunk_id = 0
    # Step size is how far we move the start index for the next chunk
    step = chunk_size - overlap

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # Manually chunk using string slicing
        start = 0
        text_length = len(text)
        
        # Continue creating chunks as long as the start index is within the text
        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Store chunk info in a dictionary
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_num + 1,  # 1-indexed page number
                "text": chunk_text
            })
            
            chunk_id += 1
            start += step

    doc.close()
    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_reader.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    
    print("-" * 60)
    all_chunks = read_and_chunk_pdf(pdf_path, chunk_size=500, overlap=50)
    print(f"Total chunks created: {len(all_chunks)}")
    print("-" * 60)
    
    print("Displaying the first 5 chunks:\n")
    for chunk in all_chunks[:5]:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page Number: {chunk['page_number']}")
        print(f"Text Length: {len(chunk['text'])} characters")
        
        # Show exactly the extracted text
        print(f"Text: {repr(chunk['text'])}")
        print("-" * 60)
