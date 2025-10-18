"""
PDF processing module for text extraction and chunking.
"""
from typing import List
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file.

        Args:
            pdf_file: Uploaded PDF file object

        Returns:
            Extracted text as string
        """
        try:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()

            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

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

