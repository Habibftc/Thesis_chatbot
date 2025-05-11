import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client with environment variable
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Process uploaded PDF files and store in Chroma
def process_documents(uploaded_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_paths = []
        for filename, file in uploaded_files.items():
            path = os.path.join(temp_dir, filename)  # ✅ fixed here
            with open(path, "wb") as f:
                f.write(file.read())  # also make sure you're using .read()
            pdf_paths.append(path)

        documents = []
        for path in pdf_paths:
            loader = PDFPlumberLoader(path)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        return vector_store


# Create retriever from stored vector DB
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

# Ask question to the model using retrieved context
def ask_question(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

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
