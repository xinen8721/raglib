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
    page_title="RAG Knowledge Base",
    page_icon="📚",
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


def process_pdf(pdf_file, processor: PDFProcessor):
    """Process uploaded PDF file."""
    try:
        with st.spinner("Processing PDF..."):
            # Process PDF
            documents = processor.process_pdf(pdf_file, pdf_file.name)
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

        # Create prompt template
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

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
        value=0.7,
        step=0.1,
        key="temperature"
    )

    st.divider()

    # Document Processing Settings
    st.subheader("Document Processing")

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks for processing"
    )

    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between consecutive chunks"
    )

    st.number_input(
        "Number of Chunks to Retrieve",
        min_value=1,
        max_value=10,
        value=4,
        key="num_chunks",
        help="Number of relevant chunks to use for answering"
    )

    st.divider()

    # PDF Upload
    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to create a knowledge base"
    )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            process_pdf(uploaded_file, processor)

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
st.title("📚 RAG Knowledge Base")
st.markdown("Upload a PDF document and ask questions about its content!")

# Check for API keys
if embedding_type == "openai" or llm_provider == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")

# Display status
if not st.session_state.documents_processed:
    st.info("👈 Please upload and process a PDF document to get started.")
else:
    st.success(f"✅ Document processed! Ask questions below.")

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
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.documents_processed:
        st.error("Please upload and process a PDF document first!")
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
    <small>Built with Streamlit, LangChain, and ChromaDB |
    Switch between OpenAI and Ollama models in the sidebar</small>
    </div>
    """,
    unsafe_allow_html=True
)

