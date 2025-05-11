import os
import tempfile
from typing import Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process_documents(uploaded_files: Dict[str, bytes], 
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> Optional[FAISS]:
    """
    Process uploaded PDF files and store embeddings in FAISS vector store.
    
    Args:
        uploaded_files: Dictionary of {filename: file_bytes}
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        FAISS vector store instance or None if error occurs
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to temp directory
            pdf_paths = []
            for filename, file in uploaded_files.items():
                path = os.path.join(temp_dir, filename)
                with open(path, "wb") as f:
                    f.write(file.read())
                pdf_paths.append(path)

            # Load and split documents
            documents = []
            for path in pdf_paths:
                loader = PDFPlumberLoader(path)
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            splits = text_splitter.split_documents(documents)

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name=DEFAULT_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )

            vector_store = FAISS.from_documents(
                documents=splits,
                embedding=embeddings
            )
            
            logger.info(f"Processed {len(splits)} document chunks")
            return vector_store

    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return None

def get_retriever(vector_store: FAISS) -> Optional[FAISS.as_retriever]:
    """
    Create retriever from FAISS vector store.
    
    Args:
        vector_store: Initialized FAISS vector store
        
    Returns:
        Retriever instance or None if error occurs
    """
    try:
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        return None

def ask_question(query: str, retriever) -> str:
    """
    Ask question using retrieved context from documents.
    
    Args:
        query: User's question
        retriever: Initialized retriever
        
    Returns:
        Answer string or error message
    """
    try:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                             for doc in docs])

        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"Error processing your question: {str(e)}"