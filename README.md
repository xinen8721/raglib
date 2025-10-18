# Streamlit RAG Application

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and ChromaDB. Upload PDFs, process them into a vector database, and ask questions using multiple LLM providers - all with a beautiful, intuitive interface.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **📄 PDF Processing**: Upload and process PDF documents with intelligent text chunking
- **🤖 Multiple LLM Providers**: Support for OpenAI (GPT-3.5, GPT-4) and Ollama (local models like Mistral, Llama2)
- **🔢 Configurable Embeddings**: Choose between OpenAI, Sentence Transformers, or Ollama embeddings
- **💾 Vector Storage**: Persistent ChromaDB storage for fast retrieval
- **💬 Interactive Chat**: Conversational interface with source document references
- **⚙️ Model Switching**: Easily switch between different models and configurations
- **🔒 Privacy-First Option**: Run completely locally with Ollama (no data sent to external APIs)
- **📊 Source Attribution**: View exact passages used to generate each answer

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+ (Python 3.13 supported)
- OpenAI API key (for OpenAI models) - [Get one here](https://platform.openai.com/api-keys)
- Optional: [Ollama](https://ollama.ai) (for free local models)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/raglib.git
cd raglib
```

2. **Create virtual environment** (REQUIRED on macOS):
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
# Copy template
cp env.template .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

5. **Verify setup** (optional but recommended):
```bash
python3 verify_setup.py
```

6. **Run the application**:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📖 Usage Guide

### Basic Workflow

1. **Configure Models** (in sidebar):
   - **Embedding Type**: Choose `openai`, `sentence-transformer`, or `ollama`
   - **Embedding Model**: e.g., `text-embedding-3-small` or `mistral`
   - **LLM Provider**: Select `openai` or `ollama`
   - **LLM Model**: e.g., `gpt-3.5-turbo` or `mistral`
   - **Temperature**: `0.1` for factual accuracy, `0.7` for conversational

2. **Upload PDF**:
   - Click "Browse files" in sidebar
   - Select your PDF document
   - Click "Process PDF"
   - Wait for processing (you'll see chunk count when done)

3. **Ask Questions**:
   - Type your question in the chat input
   - View AI-generated answers
   - Expand "View Sources" to see exact passages used

### Understanding Key Settings

#### Temperature (0.0 - 1.0)
Controls response randomness and creativity:
- **0.0 - 0.3**: Factual, accurate, minimal hallucination ⭐ **Recommended for RAG**
- **0.4 - 0.6**: Balanced
- **0.7 - 1.0**: Creative, conversational (higher hallucination risk)

#### Chunk Size
Size of text segments for processing:
- **500-800**: Technical docs, structured content
- **1000-1200**: General purpose ⭐ **Default**
- **1200-1500**: Research papers, dense content
- **1500-2000**: Legal documents, long clauses

#### Chunk Overlap
Text overlap between consecutive chunks (prevents splitting important info):
- Rule of thumb: **15-20% of chunk size**
- Default: 200 (good for chunk size 1000)

#### Chunks to Retrieve
**NOT the total chunks in your document!** This is the number of most relevant passages sent to the LLM per question.
- **2-3**: Very specific, factual queries
- **4-5**: Balanced ⭐ **Default**
- **6-8**: Complex questions needing multiple perspectives

**Example**: A 200-chunk document with "retrieve 4" means each query finds the 4 most relevant chunks from all 200, ignoring the rest.

---

## 💡 Examples & Best Practices

### Example 1: Research Paper Analysis

**Recommended Settings:**
- Embedding: `text-embedding-3-small` (OpenAI)
- LLM: `gpt-4` (better comprehension)
- Chunk Size: `1200`
- Chunks to Retrieve: `5-6`
- Temperature: `0.2`

**Example Questions:**
```
• What is the main research question addressed in this paper?
• What methodology did the authors use?
• What were the key findings and conclusions?
• What are the limitations mentioned by the authors?
```

### Example 2: Technical Documentation

**Recommended Settings:**
- Embedding: `all-MiniLM-L6-v2` (Sentence Transformer)
- LLM: `gpt-3.5-turbo` or `mistral`
- Chunk Size: `800`
- Chunks to Retrieve: `3-4`
- Temperature: `0.1`

**Example Questions:**
```
• How do I configure the authentication settings?
• What are the API rate limits?
• What error codes can be returned?
```

### Example 3: Using Ollama (100% Free & Private)

**Install Ollama:**
```bash
# Download from https://ollama.ai

# Pull models
ollama pull mistral
ollama pull nomic-embed-text  # Specialized for embeddings
```

**Recommended Settings:**
- Embedding: `nomic-embed-text` (Ollama) or `mistral`
- LLM: `mistral` (Ollama)
- Chunk Size: `1200` (no API costs!)
- Chunks to Retrieve: `5`
- Temperature: `0.2`

**Benefits:**
- ✅ Completely free
- ✅ Runs locally (full privacy)
- ✅ Works offline
- ✅ No API rate limits

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:
```bash
# Required for OpenAI models
OPENAI_API_KEY=sk-your-key-here

# Optional: Custom Ollama URL (default: http://localhost:11434)
# OLLAMA_BASE_URL=http://localhost:11434
```

### Supported Models

#### LLM Providers
- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-4o`
- **Ollama**: `llama2`, `mistral`, `phi`, `neural-chat`, `codellama`, and any locally installed models

#### Embedding Models
- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Sentence Transformers**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2` (local, free)
- **Ollama**: `mistral`, `llama2`, `nomic-embed-text` (local, free)

---

## 📁 Project Structure

```
raglib/
├── app.py                      # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF text extraction and chunking
│   ├── embeddings.py           # Configurable embedding models
│   ├── llm_handler.py          # Multi-provider LLM support
│   └── vector_store.py         # ChromaDB integration
├── requirements.txt            # Python dependencies
├── env.template               # Environment variables template
├── .gitignore                 # Git ignore file
├── verify_setup.py            # Setup verification script
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

### "OpenAI API key not found"
- Ensure `.env` file exists with valid `OPENAI_API_KEY`
- Check the key hasn't expired and has credits
- Try: `export OPENAI_API_KEY=sk-...` before running

### "Ollama not available"
- Check if Ollama is running: `ollama list`
- Verify base URL (default: `http://localhost:11434`)
- Install models: `ollama pull mistral`
- Restart Streamlit app after installing models

### "Error processing PDF"
- Ensure PDF is not encrypted or password-protected
- Try a different PDF to verify system works
- Check file isn't corrupted
- Click "Reset All" in sidebar to start fresh

### ChromaDB Issues
- Delete `chroma_db` directory to reset: `rm -rf chroma_db`
- Ensure sufficient disk space
- Try reprocessing the document

### Python Version Issues
- Use Python 3.8 or higher (3.13 tested and supported)
- On macOS: use `python3` instead of `python`
- Ensure virtual environment is activated

### Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- For Python 3.13 compatibility issues with tiktoken: Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`

---

## 🎯 Tips for Best Results

### For Maximum Accuracy (No Hallucination)
```
Temperature: 0.0 - 0.1
Chunks to Retrieve: 4-5
LLM: GPT-4 or Mistral
```

### For Speed
```
LLM: GPT-3.5-turbo or Mistral (local)
Embedding: Sentence Transformers
Chunks to Retrieve: 3
```

### For Privacy
```
Embedding: Sentence Transformers or Ollama
LLM: Ollama (Mistral/Llama2)
Everything runs locally!
```

### Document-Specific Settings
- **Research Papers**: Chunk 1200-1500, Retrieve 5-6
- **Technical Docs**: Chunk 600-800, Retrieve 3-4
- **Legal Documents**: Chunk 1500-2000, Retrieve 6-8
- **Books/Long-form**: Chunk 1000-1200, Retrieve 4-5
- **Meeting Notes**: Chunk 500-700, Retrieve 3-4

---

## 🔧 Advanced Usage

### Question Optimization
❌ **Bad**: "Tell me about X"
✅ **Good**: "What does the document say about X's impact on Y?"

❌ **Bad**: "Summarize everything"
✅ **Good**: "Summarize the key findings in the methodology section"

### Multi-step Analysis
```
1. "What is the main topic of this document?"
2. "What evidence supports the main argument?"
3. "What are the counterarguments mentioned?"
4. "What conclusions are drawn?"
```

### Using Source References
- Always check "View Sources" to verify answers
- Source documents show exact passages used
- Metadata includes chunk IDs for debugging

---

## 📊 Performance Considerations

### Token Usage (for OpenAI)
```
Total tokens ≈ (chunk_size × chunks_to_retrieve) + question + response
```
- Keep under model limits:
  - GPT-3.5: ~4K tokens
  - GPT-4: ~8K tokens
  - GPT-4-turbo: ~128K tokens

### Processing Time
- First PDF processing: Slower (creating embeddings)
- Subsequent queries: Fast (using stored embeddings)
- Local models (Ollama): Slower inference, but no API latency

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

MIT License - feel free to use this project for any purpose.

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [LangChain](https://langchain.com/) - LLM orchestration
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - LLM provider
- [Ollama](https://ollama.ai/) - Local LLM provider

---

## 📧 Support

Having issues? Check:
1. Run `python3 verify_setup.py` to diagnose problems
2. Check the Troubleshooting section above
3. Review your `.env` file configuration
4. Ensure all dependencies are installed in virtual environment

---

**Happy RAG-ing! 🚀**
