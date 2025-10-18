"""
Configurable embedding models module.
"""
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
import os


class EmbeddingManager:
    """Manages different embedding model providers."""

    OPENAI_MODELS = {
        "text-embedding-3-small": "text-embedding-3-small",
        "text-embedding-3-large": "text-embedding-3-large",
        "text-embedding-ada-002": "text-embedding-ada-002"
    }

    SENTENCE_TRANSFORMER_MODELS = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
    }

    OLLAMA_MODELS = {
        "mistral": "mistral",
        "llama2": "llama2",
        "nomic-embed-text": "nomic-embed-text"
    }

    @staticmethod
    def get_embedding_model(model_type: str, model_name: str, api_key: Optional[str] = None):
        """
        Get the specified embedding model.

        Args:
            model_type: Type of embedding ("openai", "sentence-transformer", or "ollama")
            model_name: Name of the specific model
            api_key: Optional API key for OpenAI

        Returns:
            Embedding model instance
        """
        if model_type == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")

            return OpenAIEmbeddings(
                model=EmbeddingManager.OPENAI_MODELS.get(model_name, model_name),
                openai_api_key=api_key
            )

        elif model_type == "sentence-transformer":
            model_path = EmbeddingManager.SENTENCE_TRANSFORMER_MODELS.get(
                model_name,
                model_name
            )
            return HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        elif model_type == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddings(
                model=EmbeddingManager.OLLAMA_MODELS.get(model_name, model_name),
                base_url=base_url
            )

        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")

    @staticmethod
    def get_available_models(model_type: str) -> list:
        """
        Get list of available models for a given type.

        Args:
            model_type: Type of embedding ("openai", "sentence-transformer", or "ollama")

        Returns:
            List of model names
        """
        if model_type == "openai":
            return list(EmbeddingManager.OPENAI_MODELS.keys())
        elif model_type == "sentence-transformer":
            return list(EmbeddingManager.SENTENCE_TRANSFORMER_MODELS.keys())
        elif model_type == "ollama":
            return list(EmbeddingManager.OLLAMA_MODELS.keys())
        else:
            return []

