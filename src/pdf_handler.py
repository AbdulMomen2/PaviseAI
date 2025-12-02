# PDF Handler: Extract Text/Pages from Books & Integrate to RAG/Neo4j
# =====================================================
import fitz  # PyMuPDF
from typing import List
from src.config import PDF_FILE_PATHS, PDF_SEARCH_MODE, PDF_PAGE_RANGE

class PDFHandler:
    """
    PDF Module: Load books/attachments, extract text/pages, search relevant sections.
    - Extracts full text or page ranges.
    - Searches via keyword/regex (integrates with RAG corpus).
    - Outputs chunks for Neo4j/RAG (e.g., medical book chapters).
    """
    def __init__(self):
        print("✅ PDF Handler initialized (PyMuPDF for Book Extraction)")

    def load_pdf(self, pdf_path: str) -> fitz.Document:
        """Load PDF book."""
        try:
            doc = fitz.open(pdf_path)
            print(f"📖 Loaded PDF: {pdf_path} ({doc.page_count} pages)")
            return doc
        except Exception as e:
            print(f"⚠️ PDF Load Error: {e}")
            return None

    def extract_full_text(self, pdf_path: str) -> List[str]:
        """Extract all text from PDF book."""
        doc = self.load_pdf(pdf_path)
        if not doc:
            return []
        texts = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            texts.append(f"Page {page_num+1}: {text[:500]}...")  # Chunk by page
        doc.close()
        return texts

    def extract_relevant_pages(self, pdf_path: str, query: str, mode: str = PDF_SEARCH_MODE, page_range: str = PDF_PAGE_RANGE) -> List[str]:
        """
        Search & Extract Relevant Pages/Chunks from PDF Book.
        - Mode: 'keyword' (simple match) or 'regex'.
        - Page Range: e.g., '1-50' or '1,3,5-7'.
        - Returns: Top relevant text chunks for RAG/Neo4j.
        """
        doc = self.load_pdf(pdf_path)
        if not doc:
            return []
        
        # Parse page range
        pages = self._parse_page_range(page_range, doc.page_count)
        
        relevant_chunks = []
        for page_num in pages:
            page = doc.load_page(page_num)
            text = page.get_text().lower()
            
            if mode == "keyword":
                if any(word in text for word in query.lower().split()):
                    relevant_chunks.append(page.get_text())
            elif mode == "regex":
                import re
                if re.search(query.lower(), text):
                    relevant_chunks.append(page.get_text())
        
        doc.close()
        print(f"🔍 Extracted {len(relevant_chunks)} relevant chunks from PDF for query: '{query}'")
        return relevant_chunks

    def _parse_page_range(self, range_str: str, total_pages: int) -> List[int]:
        """Parse '1-50,52,55-60' to list of page nums."""
        pages = set()
        parts = range_str.split(',')
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
        return sorted(list(pages)[:total_pages])  # Clamp to total

    def integrate_to_corpus(self, pdf_paths: List[str], query: str) -> List[str]:
        """Extract & Return PDF Chunks for RAG Corpus."""
        corpus_chunks = []
        for pdf_path in pdf_paths:
            chunks = self.extract_relevant_pages(pdf_path, query)
            corpus_chunks.extend([f"PDF Book [{pdf_path}]: {chunk}" for chunk in chunks])
        return corpus_chunks