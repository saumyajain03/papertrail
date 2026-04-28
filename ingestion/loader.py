import sys
import fitz  # PyMuPDF

def load_pdf(file_path: str) -> list:
    """Reads a PDF and returns a list of dictionaries with page number and extracted text."""
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)

    print(f"Loaded PDF: {file_path}")
    print(f"Total pages: {len(doc)}")
    
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        pages.append({
            "page_number": page_num + 1,  # 1-indexed
            "text": text
        })

    doc.close()
    return pages
