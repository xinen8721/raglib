"""
ChromaDB vector store integration.
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import chromadb
from chromadb.config import Settings
import os


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize vector store manager.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        self.vector_store = None

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

    def create_vector_store(self, documents: List[Document],
                          embeddings: Embeddings,
                          collection_name: str = "documents") -> Chroma:
        """
        Create a new vector store from documents.

        Args:
            documents: List of Document objects to embed
            embeddings: Embedding model to use
            collection_name: Name for the ChromaDB collection

        Returns:
            Chroma vector store instance
        """
        try:
            # Create ChromaDB client with settings
            client_settings = Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )

            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory,
                client_settings=client_settings
            )

            return self.vector_store

        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    def load_vector_store(self, embeddings: Embeddings,
                         collection_name: str = "documents") -> Optional[Chroma]:
        """
        Load an existing vector store.

        Args:
            embeddings: Embedding model (must match the one used to create)
            collection_name: Name of the ChromaDB collection

        Returns:
            Chroma vector store instance or None if not found
        """
        try:
            client_settings = Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )

            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
                client_settings=client_settings
            )

            return self.vector_store

        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")

        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, search_kwargs: dict = None):
        """
        Get a retriever instance for the vector store.

        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 4})

        Returns:
            Retriever instance
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")

        if search_kwargs is None:
            search_kwargs = {"k": 4}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def delete_collection(self, collection_name: str = "documents"):
        """
        Delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete
        """
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")

