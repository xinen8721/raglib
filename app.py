"""
Streamlit RAG Application - Main Interface
"""
import streamlit as st
from dotenv import load_dotenv
import os
from typing import Optional

from utils.pdf_processor import PDFProcessor
from utils.embeddings import EmbeddingManager
from utils.llm_handler import LLMHandler
from utils.vector_store import VectorStoreManager

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant - IRAC Analysis",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "current_filename" not in st.session_state:
    st.session_state.current_filename = None


def initialize_components():
    """Initialize all necessary components based on sidebar configuration."""
    try:
        # Get embedding model
        embedding_model = EmbeddingManager.get_embedding_model(
            model_type=st.session_state.embedding_type,
            model_name=st.session_state.embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Get LLM
        llm = LLMHandler.get_llm(
            provider=st.session_state.llm_provider,
            model_name=st.session_state.llm_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=st.session_state.temperature
        )

        return embedding_model, llm

    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None


def process_pdf(pdf_file, processor: PDFProcessor, use_ocr: bool = False):
    """Process uploaded PDF file."""
    try:
        with st.spinner("Processing PDF..." + (" with OCR" if use_ocr else "")):
            # Extract text with optional OCR
            text = processor.extract_text_from_pdf(pdf_file, use_ocr=use_ocr)

            # Chunk text
            documents = processor.chunk_text(text, metadata={"source": pdf_file.name})
            st.session_state.current_filename = pdf_file.name

            # Get embedding model
            embedding_model, _ = initialize_components()

            if embedding_model is None:
                return False

            # Create vector store
            vector_manager = VectorStoreManager()
            vector_store = vector_manager.create_vector_store(
                documents=documents,
                embeddings=embedding_model,
                collection_name="current_session"
            )

            st.session_state.vector_store = vector_manager
            st.session_state.documents_processed = True

            st.success(f"✅ Processed {len(documents)} chunks from {pdf_file.name}")
            return True

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False


def get_rag_response(query: str):
    """Get response from RAG pipeline."""
    try:
        # Initialize components
        embedding_model, llm = initialize_components()

        if embedding_model is None or llm is None:
            return "Error: Could not initialize components. Check your API keys."

        if st.session_state.vector_store is None:
            return "Please upload and process a PDF document first."

        # Get retriever
        retriever = st.session_state.vector_store.get_retriever(
            search_kwargs={"k": st.session_state.num_chunks}
        )

        # Retrieve documents
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(query)

        # Create IRAC method prompt template for legal analysis
        template = """You are a legal AI assistant. Analyze the question using the IRAC method (Issue, Rule, Application, Conclusion) based on the provided legal document context.

Context from Legal Document:
{context}

Legal Question: {question}

Provide your analysis in IRAC format:

**ISSUE:**
[Clearly state the legal issue or question to be resolved]

**RULE:**
[State the applicable legal rules, principles, or provisions from the document]

**APPLICATION:**
[Apply the rules to the specific facts or situation in the question, referencing specific sections of the document]

**CONCLUSION:**
[Provide a clear conclusion answering the legal question]

Please structure your response following this IRAC format strictly. If the document doesn't contain relevant information, state that clearly in the Issue section."""

        prompt = ChatPromptTemplate.from_template(template)

        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create chain using LCEL
        chain = prompt | llm | StrOutputParser()

        # Get response
        with st.spinner("Thinking..."):
            answer = chain.invoke({"context": context, "question": query})

        # Format response to match expected structure
        formatted_response = {
            "result": answer,
            "source_documents": docs
        }

        return formatted_response

    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None


# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    # Check if Ollama is available (used for both embeddings and LLM)
    ollama_available = LLMHandler.check_ollama_available()

    # Embedding Model Selection
    st.subheader("Embedding Model")

    # Add Ollama option if available
    embedding_providers = ["openai", "sentence-transformer"]
    if ollama_available:
        embedding_providers.append("ollama")

    embedding_type = st.selectbox(
        "Embedding Type",
        embedding_providers,
        key="embedding_type",
        help="Ollama appears if Ollama is running locally"
    )

    embedding_models = EmbeddingManager.get_available_models(embedding_type)

    st.selectbox(
        "Model",
        embedding_models,
        key="embedding_model"
    )

    st.divider()

    # LLM Provider Selection
    st.subheader("Language Model")

    providers = ["openai"]
    if ollama_available:
        providers.append("ollama")

    llm_provider = st.selectbox(
        "Provider",
        providers,
        key="llm_provider",
        help="Ollama must be running locally to appear as an option"
    )

    llm_models = LLMHandler.get_available_models(llm_provider)
    st.selectbox(
        "Model",
        llm_models,
        key="llm_model"
    )

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        key="temperature",
        help="Low temperature for precise legal analysis (0.1-0.3 recommended)"
    )

    st.divider()

    # Document Processing Settings
    st.subheader("📄 Legal Document Settings")

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=3000,
        value=1500,
        step=100,
        help="Larger chunks for legal clauses (recommended: 1500-2000)"
    )

    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=300,
        step=50,
        help="Higher overlap for legal context (recommended: 250-400)"
    )

    st.number_input(
        "Number of Chunks to Retrieve",
        min_value=1,
        max_value=12,
        value=6,
        key="num_chunks",
        help="More chunks for comprehensive legal analysis (recommended: 6-8)"
    )

    st.divider()

    # PDF Upload
    st.subheader("⚖️ Upload Legal Document")

    # OCR Option
    use_ocr = st.checkbox(
        "📸 Enable OCR (for scanned documents)",
        value=False,
        help="Enable this for scanned contracts, photocopies, or image-based PDFs. Slower but can read text from images."
    )

    uploaded_file = st.file_uploader(
        "Upload Legal Document (PDF)",
        type="pdf",
        help="Upload contracts, statutes, case law, regulations, or any legal document"
    )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            process_pdf(uploaded_file, processor, use_ocr=use_ocr)

    # Show current document
    if st.session_state.current_filename:
        st.info(f"📁 Current: {st.session_state.current_filename}")

    st.divider()

    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Reset everything
    if st.button("Reset All", help="Clear all data and start fresh"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.documents_processed = False
        st.session_state.current_filename = None
        st.rerun()


# Main content area
st.title("⚖️ Legal AI Assistant")
st.markdown("Upload legal documents and get IRAC method analysis for your legal questions")

# Add IRAC method explanation
with st.expander("ℹ️ What is IRAC Method?"):
    st.markdown("""
    **IRAC** is a legal analysis framework used by attorneys and law students:

    - **I**ssue: What is the legal question?
    - **R**ule: What law/rule applies?
    - **A**pplication: How does the rule apply to the facts?
    - **C**onclusion: What is the answer?

    This AI assistant will analyze your legal documents using this structured approach.
    """)

# Check for API keys
if embedding_type == "openai" or llm_provider == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

# Display status
if not st.session_state.documents_processed:
    st.info("👈 Please upload a legal document (contract, statute, case law, etc.) to begin analysis.")
else:
    st.success(f"✅ Legal document processed! Ask your legal questions below for IRAC analysis.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📎 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source.page_content[:300] + "...")
                    st.markdown(f"*Metadata: {source.metadata}*")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a legal question (e.g., 'What are the termination clauses?', 'What are the liability provisions?')"):
    if not st.session_state.documents_processed:
        st.error("Please upload and process a legal document first!")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get RAG response
        response = get_rag_response(prompt)

        if response:
            # Add assistant message to chat
            assistant_message = {
                "role": "assistant",
                "content": response["result"],
                "sources": response.get("source_documents", [])
            }
            st.session_state.messages.append(assistant_message)

            with st.chat_message("assistant"):
                st.markdown(response["result"])

                # Display sources
                if response.get("source_documents"):
                    with st.expander("📎 View Sources"):
                        for i, source in enumerate(response["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source.page_content[:300] + "...")
                            st.markdown(f"*Metadata: {source.metadata}*")
                            st.divider()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>⚖️ Legal AI Assistant | IRAC Method Analysis | Built with Streamlit, LangChain, and ChromaDB<br>
    <strong>Disclaimer:</strong> This tool provides AI-generated legal analysis for informational purposes only.
    It is not a substitute for professional legal advice. Consult a qualified attorney for legal matters.</small>
    </div>
    """,
    unsafe_allow_html=True
)

