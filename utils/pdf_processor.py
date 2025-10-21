"""
PDF processing module for text extraction and chunking with OCR support.
"""
from typing import List
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os

# OCR imports (optional, will check if available)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class PDFProcessor:
    """Handles PDF text extraction and chunking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_file, use_ocr: bool = False) -> str:
        """
        Extract text from uploaded PDF file with optional OCR support.

        Args:
            pdf_file: Uploaded PDF file object
            use_ocr: If True, use OCR for scanned/image-based PDFs

        Returns:
            Extracted text as string
        """
        try:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""
            ocr_pages = []

            # Try regular text extraction first
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()

                # Check if page has little or no text (likely scanned)
                if use_ocr and OCR_AVAILABLE and (not page_text.strip() or len(page_text.strip()) < 50):
                    ocr_pages.append(page_num)
                    page_text = ""  # Will OCR later

                text += page_text + "\n"

            # If OCR is enabled and we found pages needing OCR
            if use_ocr and ocr_pages and OCR_AVAILABLE:
                ocr_text = self._ocr_pdf_pages(pdf_file, ocr_pages)
                # Replace or append OCR text
                if ocr_text:
                    text += "\n" + ocr_text

            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def _ocr_pdf_pages(self, pdf_file, page_numbers: List[int]) -> str:
        """
        Extract text from specific PDF pages using OCR.

        Args:
            pdf_file: Uploaded PDF file object
            page_numbers: List of page numbers to OCR (0-indexed)

        Returns:
            Extracted text from OCR
        """
        if not OCR_AVAILABLE:
            return ""

        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
                pdf_file.seek(0)  # Reset file pointer

            ocr_text = ""

            # Process each page that needs OCR
            for page_num in page_numbers:
                try:
                    # Convert PDF page to image
                    images = convert_from_path(
                        tmp_path,
                        first_page=page_num + 1,
                        last_page=page_num + 1,
                        dpi=300  # Higher DPI for better OCR accuracy
                    )

                    if images:
                        # Run OCR on the image
                        page_text = pytesseract.image_to_string(images[0], lang='eng')
                        ocr_text += f"\n--- Page {page_num + 1} (OCR) ---\n{page_text}\n"

                except Exception as e:
                    print(f"OCR failed for page {page_num + 1}: {str(e)}")
                    continue

            # Clean up temporary file
            os.unlink(tmp_path)

            return ocr_text

        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            return ""

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Create Document objects with metadata
        documents = [
            Document(page_content=chunk, metadata={**metadata, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]

        return documents

    def process_pdf(self, pdf_file, filename: str = None) -> List[Document]:
        """
        Complete PDF processing pipeline: extract and chunk.

        Args:
            pdf_file: Uploaded PDF file object
            filename: Optional filename for metadata

        Returns:
            List of Document objects ready for embedding
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_file)

        # Prepare metadata
        metadata = {"source": filename or "unknown"}

        # Chunk text
        documents = self.chunk_text(text, metadata)

        return documents

