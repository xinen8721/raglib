#!/usr/bin/env python3
# Run with: python3 verify_setup.py (macOS/Linux) or python verify_setup.py (Windows)
"""
Setup verification script for RAG application.
Run this script to verify your environment is correctly configured.
"""
import sys
import os

def check_imports():
    """Check if all required packages are installed."""
    print("Checking Python packages...")
    required_packages = [
        ("streamlit", "streamlit"),
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_openai", "langchain-openai"),
        ("chromadb", "chromadb"),
        ("pypdf", "pypdf"),
        ("sentence_transformers", "sentence-transformers"),
        ("dotenv", "python-dotenv"),
        ("openai", "openai"),
        ("requests", "requests")
    ]

    missing = []
    for package, install_name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {install_name}")
        except ImportError:
            print(f"  ✗ {install_name} - MISSING")
            missing.append(install_name)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✅ All packages installed!\n")
    return True


def check_env_file():
    """Check if .env file exists and has required keys."""
    print("Checking environment configuration...")

    if not os.path.exists(".env"):
        print("  ⚠️  .env file not found")
        print("  Create one with: cp env.template .env")
        print("  Then add your OpenAI API key\n")
        return False

    print("  ✓ .env file exists")

    # Load .env
    from dotenv import load_dotenv
    load_dotenv()

    # Check for OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  ⚠️  OPENAI_API_KEY not set in .env")
        print("  Add your key: OPENAI_API_KEY=sk-...")
        print("  (Required for OpenAI models and embeddings)\n")
        return False

    if openai_key == "your_openai_key_here":
        print("  ⚠️  OPENAI_API_KEY still has placeholder value")
        print("  Replace with your actual API key\n")
        return False

    print("  ✓ OPENAI_API_KEY is set")
    print("✅ Environment configured!\n")
    return True


def check_ollama():
    """Check if Ollama is available (optional)."""
    print("Checking Ollama (optional)...")

    try:
        import requests
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/tags", timeout=2)

        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"  ✓ Ollama is running")
            if models:
                print(f"  ✓ Available models: {', '.join(models)}")
            else:
                print("  ⚠️  No models installed. Install with: ollama pull llama2")
            print("✅ Ollama ready!\n")
            return True
        else:
            print("  ⚠️  Ollama not responding")
            print("  Install from: https://ollama.ai")
            print("  (Optional - OpenAI models will still work)\n")
            return False

    except Exception as e:
        print("  ⚠️  Ollama not available")
        print("  Install from: https://ollama.ai")
        print("  (Optional - OpenAI models will still work)\n")
        return False


def test_basic_functionality():
    """Test basic functionality of the utilities."""
    print("Testing basic functionality...")

    try:
        # Test core imports
        import streamlit
        import langchain
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        print("  ✓ Core packages importable")

        # Test PDF processor
        from utils.pdf_processor import PDFProcessor
        processor = PDFProcessor()
        print("  ✓ PDF processor initialized")

        # Test embedding manager
        from utils.embeddings import EmbeddingManager
        openai_models = EmbeddingManager.get_available_models("openai")
        print(f"  ✓ Embedding manager: {len(openai_models)} OpenAI models available")

        # Test LLM handler
        from utils.llm_handler import LLMHandler
        llm_models = LLMHandler.get_available_models("openai")
        print(f"  ✓ LLM handler: {len(llm_models)} OpenAI models available")

        # Test vector store manager
        from utils.vector_store import VectorStoreManager
        vector_manager = VectorStoreManager(persist_directory="./test_chroma_db")
        print("  ✓ Vector store manager initialized")

        # Clean up test directory
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")

        print("✅ All utilities working!\n")
        return True

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        print("❌ Functionality test failed\n")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("RAG Application Setup Verification")
    print("=" * 60 + "\n")

    checks = [
        ("Package Installation", check_imports),
        ("Environment Configuration", check_env_file),
        ("Ollama Availability", check_ollama),
        ("Basic Functionality", test_basic_functionality)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} check failed with error: {str(e)}\n")
            results.append(False)

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if all(results[:2]):  # First two are critical
        print("✅ Your environment is ready to run the RAG application!")
        print("\nStart the app with: streamlit run app.py")
    elif results[0]:
        print("⚠️  Packages installed but environment needs configuration")
        print("\nFollow the setup instructions in QUICKSTART.md")
    else:
        print("❌ Setup incomplete. Please install required packages.")
        print("\nRun: pip install -r requirements.txt")

    if not results[2]:
        print("\nNote: Ollama is optional. You can use OpenAI models without it.")

    print("\nFor detailed instructions, see QUICKSTART.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

