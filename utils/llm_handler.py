"""
Multi-provider LLM handler for OpenAI and Ollama.
"""
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import os
import requests


class LLMHandler:
    """Handles multiple LLM providers."""

    OPENAI_MODELS = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o"
    ]

    OLLAMA_DEFAULT_MODELS = [
        "llama2",
        "mistral",
        "phi",
        "neural-chat",
        "codellama"
    ]

    @staticmethod
    def get_llm(provider: str, model_name: str, api_key: Optional[str] = None,
                temperature: float = 0.7, **kwargs):
        """
        Get the specified LLM instance.

        Args:
            provider: LLM provider ("openai" or "ollama")
            model_name: Name of the model
            api_key: Optional API key for OpenAI
            temperature: Temperature for generation
            **kwargs: Additional arguments for the LLM

        Returns:
            LLM instance
        """
        if provider == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")

            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                **kwargs
            )

        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return Ollama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                **kwargs
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """
        Get list of available models for a provider.

        Args:
            provider: LLM provider ("openai" or "ollama")

        Returns:
            List of model names
        """
        if provider == "openai":
            return LLMHandler.OPENAI_MODELS

        elif provider == "ollama":
            # Try to detect installed Ollama models
            try:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                response = requests.get(f"{base_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    return models if models else LLMHandler.OLLAMA_DEFAULT_MODELS
            except:
                pass

            return LLMHandler.OLLAMA_DEFAULT_MODELS

        return []

    @staticmethod
    def check_ollama_available() -> bool:
        """
        Check if Ollama is running and available.

        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

