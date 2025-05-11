import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from groq import APITimeoutError, APIError
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Groq client with environment variable
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process_documents(uploaded_files):
    """Process uploaded PDF files and store embeddings in FAISS vector store"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_paths = []
            for filename, file in uploaded_files.items():
                path = os.path.join(temp_dir, filename)
                with open(path, "wb") as f:
                    f.write(file.read())
                pdf_paths.append(path)

            documents = []
            for path in pdf_paths:
                loader = PDFPlumberLoader(path)
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, 
                chunk_overlap=150
            )
            splits = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )

            vector_store = FAISS.from_documents(
                documents=splits,
                embedding=embeddings
            )
            
            # Save the FAISS index
            vector_store.save_local("faiss_index")
            return vector_store
            
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise Exception("Failed to process documents. Please check the files and try again.")

def get_retriever():
    """Create retriever from stored FAISS vector DB"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        vector_store = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

def ask_question(query, retriever):
    """Ask question to the model using retrieved context with retry mechanism"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )

            return response.choices[0].message.content
            
        except APITimeoutError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "The request timed out. Please try again later."
        except APIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Error processing your question: {str(e)}"
    
    return "Sorry, I couldn't process your request after multiple attempts."