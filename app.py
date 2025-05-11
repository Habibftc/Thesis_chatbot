import streamlit as st
from model import get_response
from pdf_model import process_documents, get_retriever, ask_question
from image_model import encode_image, analyze_image
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style> 
        .user { background-color: #2c2f36; padding: 10px; border-radius: 10px; }
        .bot { background-color: #353a42; padding: 10px; border-radius: 10px; }
        .container { width: 60%; margin: auto; }
        .message { margin-bottom: 10px; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #4a6fa5 0%, #3a5a8a 100%);
            color: white;
        }
        .sidebar-title { color: white !important; }
        .sidebar-section { padding: 10px; border-radius: 5px; margin-bottom: 15px; }
        .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar Navigation
with st.sidebar:
    st.markdown('<h1 class="sidebar-title">🔍 AI Assistant</h1>', unsafe_allow_html=True)

    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["💬 Chat", "📄 PDF Q&A", "🖼️ Image Analysis"],
        index=0,
        key="mode_selector"
    )

    # Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7)
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF Mode
    if mode == "📄 PDF Q&A":
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📄 PDF Options")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process PDFs"):
                    with st.spinner("Processing PDFs..."):
                        vector_store = process_documents({f.name: f for f in uploaded_files})
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.retriever = get_retriever(vector_store)
                            st.session_state.pdf_processed = True
                            st.success("PDFs processed successfully!")
                        else:
                            st.error("Failed to process PDFs")
            with col2:
                if st.button("Clear PDFs"):
                    st.session_state.pdf_processed = False
                    st.session_state.vector_store = None
                    st.session_state.messages = []
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Image Mode
    elif mode == "🖼️ Image Analysis":
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Image Options")
        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.session_state.uploaded_image = encode_image(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Clear Image"):
                st.session_state.uploaded_image = None
                st.session_state.messages = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Clear conversation button
    if st.session_state.messages:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Main Title
st.markdown("<h1 style='text-align: center;'>🤖 AI Assistant</h1>", unsafe_allow_html=True)

# Chat History
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "bot"
    if message.get("type") == "image":
        st.image(message["content"], caption='Uploaded Image', use_column_width=True)
    else:
        st.markdown(f"<div class='message {role}'>{message['content']}</div>", unsafe_allow_html=True)

# Chat Mode
if mode == "💬 Chat":
    user_input = st.chat_input("Type your message here...", key="chat_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"<div class='message user'>{user_input}</div>", unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            response = get_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": response})
        st.markdown(f"<div class='message bot'>{response}</div>", unsafe_allow_html=True)

# PDF Mode
elif mode == "📄 PDF Q&A":
    if st.session_state.get('pdf_processed', False):
        question = st.chat_input("Ask about the PDFs...", key="pdf_question")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            st.markdown(f"<div class='message user'>{question}</div>", unsafe_allow_html=True)

            with st.spinner("Searching documents..."):
                answer = ask_question(question, st.session_state.retriever)
            st.session_state.messages.append({"role": "bot", "content": answer})
            st.markdown(f"<div class='message bot'>{answer}</div>", unsafe_allow_html=True)
    else:
        st.info("Please upload and process PDF files first")

# Image Mode
elif mode == "🖼️ Image Analysis":
    if st.session_state.uploaded_image:
        question = st.chat_input("Ask about the image...", key="image_question")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            st.markdown(f"<div class='message user'>{question}</div>", unsafe_allow_html=True)

            with st.spinner("Analyzing image..."):
                response = analyze_image(st.session_state.uploaded_image, question)
            st.session_state.messages.append({"role": "bot", "content": response})
            st.markdown(f"<div class='message bot'>{response}</div>", unsafe_allow_html=True)
    else:
        st.info("Please upload an image first")