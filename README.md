# ⚖️ Legal AI Assistant - IRAC Method Analysis

A specialized legal document AI assistant built with Streamlit, LangChain, and ChromaDB. Upload legal documents (contracts, statutes, case law, regulations) and get structured IRAC method analysis using multiple LLM providers.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **⚖️ IRAC Method Analysis**: Structured legal reasoning (Issue, Rule, Application, Conclusion)
- **📄 Legal Document Processing**: Optimized for contracts, statutes, case law, and regulations
- **🤖 Multiple LLM Providers**: Support for OpenAI (GPT-3.5, GPT-4) and Ollama (local models)
- **🔢 Configurable Embeddings**: Choose between OpenAI, Sentence Transformers, or Ollama embeddings
- **💾 Vector Storage**: Persistent ChromaDB storage for fast legal document **retrieval**
- **💬 Legal Q&A**: Ask legal questions and get IRAC-formatted responses
- **⚙️ Optimized for Legal Docs**: Larger chunk sizes (1500) and higher overlap (300) for legal clauses
- **🔒 Privacy-First Option**: Run completely locally with Ollama for sensitive legal documents
- **📊 Source Citation**: View exact legal provisions used in analysis
- **⚠️ Professional Disclaimer**: Clear disclaimer that this is for informational purposes only

---

## 🎬 Demo

See the application in action! Upload a PDF, process it, and start asking questions with full source attribution.

![RAG Knowledge Base Demo](screenshots/demo.gif)

### What You're Seeing:
- ✅ **Legal Document Processed**: Chunks extracted from legal document
- ⚖️ **IRAC Method**: Responses structured with Issue, Rule, Application, Conclusion
- 🎯 **Legal-Optimized Settings**: Larger chunks (1500), higher overlap (300) for legal clauses
- 💬 **Legal Q&A**: Ask questions about clauses, provisions, obligations, liabilities
- 📚 **Legal Citations**: Each answer shows exact document provisions used

### Try It Yourself:
```bash
# Clone and run
git clone https://github.com/xinen8721/raglib.git
cd raglib
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.template .env
# Add your OPENAI_API_KEY to .env
streamlit run app.py
```

Upload any PDF and start asking questions in seconds! 🚀

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
   - **LLM Model**: e.g., `gpt-4` (recommended for legal) or `mistral`
   - **Temperature**: `0.2` (default - low for precise legal analysis)

2. **Upload Legal Document**:
   - Click "Browse files" in sidebar
   - Select your legal PDF (contract, statute, case law, etc.)
   - Click "Process PDF"
   - Wait for processing with legal-optimized chunking

3. **Ask Legal Questions**:
   - Type your legal question (e.g., "What are the termination clauses?")
   - Receive IRAC-formatted analysis (Issue, Rule, Application, Conclusion)
   - Expand "View Sources" to see exact legal provisions cited

### Understanding Key Settings

#### Temperature (0.0 - 1.0)
Controls response randomness and creativity:
- **0.1 - 0.3**: Precise, factual legal analysis ⭐ **Recommended for legal documents** (Default: 0.2)
- **0.4 - 0.6**: Balanced (not recommended for legal)
- **0.7 - 1.0**: Creative, conversational (⚠️ high risk for legal analysis)

#### Chunk Size
Size of text segments for processing:
- **500-800**: Technical docs, structured content
- **1000-1200**: General documents
- **1500-2000**: Legal documents, long clauses ⭐ **Default for legal** (1500)
- **2000-3000**: Complex legal contracts with lengthy provisions

#### Chunk Overlap
Text overlap between consecutive chunks (prevents splitting important legal clauses):
- Rule of thumb: **15-25% of chunk size**
- Legal documents: **250-400** (higher overlap critical for legal context)
- Default: 300 (optimized for legal chunk size 1500)

#### Chunks to Retrieve
**NOT the total chunks in your document!** This is the number of most relevant legal passages sent to the LLM per question.
- **2-3**: Very specific clause lookups
- **4-5**: Balanced for general documents
- **6-8**: Legal analysis needing comprehensive context ⭐ **Default for legal** (6)
- **8-12**: Complex legal questions spanning multiple provisions

**Example**: A 200-chunk document with "retrieve 6" means each query finds the 6 most relevant legal chunks from all 200, ignoring the rest.

---

## ⚖️ Understanding IRAC Method

**IRAC** is the standard framework for legal analysis used by attorneys, law students, and legal professionals:

### What is IRAC?

- **I**ssue: Identify the specific legal question or problem
- **R**ule: State the applicable law, regulation, or principle
- **A**pplication: Apply the rule to the specific facts
- **C**onclusion: Provide the answer or outcome

### Example IRAC Analysis

**Question**: "What are the termination provisions in this employment contract?"

**ISSUE:**
The issue is whether the employment contract contains provisions allowing for termination and under what circumstances termination may occur.

**RULE:**
According to Section 8.2 of the Employment Agreement, either party may terminate the agreement with 30 days written notice. Section 8.3 provides for immediate termination for cause, including material breach, fraud, or gross negligence.

**APPLICATION:**
The contract explicitly provides two termination mechanisms. For routine termination without cause, the 30-day notice requirement under Section 8.2 must be satisfied. However, if the employer can demonstrate cause as defined in Section 8.3 (material breach, fraud, or gross negligence), immediate termination is permitted without the 30-day notice period.

**CONCLUSION:**
The contract allows termination in two ways: (1) with 30 days written notice for any reason, or (2) immediately for cause as specifically defined in Section 8.3.

---

## 💡 Legal Document Examples & Best Practices

### Example 1: Employment Contract Analysis

**Recommended Settings:**
- Embedding: `text-embedding-3-small` (OpenAI)
- LLM: `gpt-4` (best for legal nuance)
- Chunk Size: `1500`
- Chunks to Retrieve: `6`
- Temperature: `0.2`

**Example Legal Questions:**
```
• What are the termination provisions in this contract?
• What are the non-compete obligations?
• What remedies are available for breach of contract?
• What is the governing law and jurisdiction?
• What are the confidentiality requirements?
• What are the compensation and benefits terms?
```

### Example 2: Commercial Lease Agreement

**Recommended Settings:**
- Embedding: `text-embedding-3-small` (OpenAI)
- LLM: `gpt-4`
- Chunk Size: `1800`
- Chunks to Retrieve: `7`
- Temperature: `0.2`

**Example Legal Questions:**
```
• What are the rent escalation clauses?
• What are the tenant's maintenance obligations?
• What constitutes a default under this lease?
• What are the options for lease renewal?
• What are the permitted uses of the premises?
```

### Example 3: Regulatory Compliance Document

**Recommended Settings:**
- Embedding: `text-embedding-3-large` (OpenAI - higher accuracy)
- LLM: `gpt-4`
- Chunk Size: `2000`
- Chunks to Retrieve: `8`
- Temperature: `0.1` (maximum precision)

**Example Legal Questions:**
```
• What are the reporting requirements under this regulation?
• What penalties apply for non-compliance?
• What are the exemptions or safe harbors available?
• What is the timeline for implementation?
```

### Example 4: Using Ollama for Sensitive Legal Documents (100% Private)

**Perfect for confidential legal documents that cannot be sent to external APIs!**

**Install Ollama:**
```bash
# Download from https://ollama.ai

# Pull models optimized for legal analysis
ollama pull mistral      # Good general model
ollama pull llama2       # Alternative model
ollama pull nomic-embed-text  # Best for embeddings
```

**Recommended Settings:**
- Embedding: `nomic-embed-text` (Ollama)
- LLM: `mistral` (Ollama)
- Chunk Size: `1500`
- Chunks to Retrieve: `6`
- Temperature: `0.2`

**Benefits for Legal Practice:**
- ✅ **100% Private**: No data leaves your computer
- ✅ **Client Confidentiality**: Attorney-client privilege maintained
- ✅ **Completely Free**: No per-document or per-query costs
- ✅ **No Rate Limits**: Analyze as many documents as needed
- ✅ **Works Offline**: No internet required
- ✅ **Compliant**: Meets data privacy regulations

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

## 🎯 Tips for Best Legal Analysis Results

### For Maximum Legal Accuracy (Recommended)
```
Temperature: 0.1 - 0.2
Chunk Size: 1500-1800
Chunk Overlap: 300-400
Chunks to Retrieve: 6-8
LLM: GPT-4 (best legal reasoning)
Embedding: text-embedding-3-small or text-embedding-3-large
```

### For Confidential Legal Documents (Privacy First)
```
Temperature: 0.2
Chunk Size: 1500
Chunk Overlap: 300
Chunks to Retrieve: 6
LLM: Ollama Mistral (100% local)
Embedding: Ollama nomic-embed-text (100% local)
✅ All data stays on your computer
✅ Maintains attorney-client privilege
```

### For Quick Contract Review (Speed vs. Accuracy)
```
Temperature: 0.2
Chunk Size: 1200
Chunks to Retrieve: 5
LLM: GPT-3.5-turbo
Embedding: all-MiniLM-L6-v2 (local, fast)
```

### Legal Document-Specific Settings

#### Employment Contracts
- **Chunk Size**: 1500
- **Overlap**: 300
- **Retrieve**: 6
- **Temperature**: 0.2
- **Why**: Balances clause completeness with context

#### Commercial Leases
- **Chunk Size**: 1800
- **Overlap**: 350
- **Retrieve**: 7
- **Temperature**: 0.2
- **Why**: Longer provisions, more interconnected clauses

#### Corporate Bylaws / Articles
- **Chunk Size**: 1500
- **Overlap**: 300
- **Retrieve**: 6
- **Temperature**: 0.15
- **Why**: Highly structured, cross-references common

#### Litigation Documents (Pleadings, Motions)
- **Chunk Size**: 1200
- **Overlap**: 250
- **Retrieve**: 6-8
- **Temperature**: 0.2
- **Why**: Dense arguments, need broader context

#### Regulatory / Compliance Documents
- **Chunk Size**: 2000
- **Overlap**: 400
- **Retrieve**: 8
- **Temperature**: 0.1
- **Why**: Maximum precision required, lengthy provisions

#### NDAs / Confidentiality Agreements
- **Chunk Size**: 1200
- **Overlap**: 250
- **Retrieve**: 5
- **Temperature**: 0.2
- **Why**: Shorter documents, focused scope

---

## 🔧 Advanced Legal Usage

### Legal Question Optimization

#### ❌ **Bad Legal Questions:**
- "Tell me about this contract"
- "What's in this document?"
- "Summarize everything"

#### ✅ **Good Legal Questions:**
- "What are the termination provisions in this agreement?"
- "What remedies are available for breach under Section 10?"
- "What are the indemnification obligations of each party?"
- "What events constitute a material breach?"
- "What is the statute of limitations for claims under this contract?"
- "What are the conditions precedent to the buyer's obligations?"

### Multi-step Legal Analysis
```
1. "What is the main purpose of this agreement?"
2. "What are the key obligations of each party?"
3. "What are the termination rights?"
4. "What dispute resolution mechanism is specified?"
5. "What is the governing law and jurisdiction?"
```

### Contract Drafting Review
```
1. "Are there any undefined terms in this agreement?"
2. "What notice requirements are specified?"
3. "Are there any automatic renewal clauses?"
4. "What are the limitation of liability provisions?"
5. "Are there any missing standard clauses?"
```

### Using Source References (Critical for Legal Work)
- **ALWAYS** check "View Sources" to verify legal citations
- Source documents show exact contractual language used
- Verify section numbers and clause references
- Never rely solely on AI summary for legal advice
- Use sources to quote exact contract language in memos

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

## ⚠️ Important Legal Disclaimer

**THIS TOOL IS FOR INFORMATIONAL PURPOSES ONLY**

- This AI assistant provides legal **analysis**, not legal **advice**
- AI-generated responses may contain errors or inaccuracies
- **Always verify** AI responses with original documents
- **Never rely solely** on AI for legal decision-making
- This tool does **NOT** create an attorney-client relationship
- For legal matters, **consult a qualified attorney**
- Users are responsible for verifying accuracy of all outputs
- This tool should be used as a **research aid**, not a replacement for legal counsel

**For Legal Professionals:**
- Use as a document review assistant to speed up initial analysis
- Always review and verify AI-generated IRAC analysis
- Check all cited sections and provisions in original documents
- Apply your professional judgment to all AI outputs
- Maintain professional responsibility and ethical obligations

---

## 📝 License

MIT License - feel free to use this project for any purpose. However, users assume all responsibility for the use of this tool and its outputs.

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
