import streamlit as st
from model import get_response
from pdf_model import process_documents, get_retriever, ask_question
from image import encode_image, analyze_image
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .user { background-color: #2C2F36; padding: 10px; border-radius: 10px; }
        .bot { background-color: #353A42; padding: 10px; border-radius: 10px; }
        .container { width: 60%; margin: auto; }
        .message { margin-bottom: 10px; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #4a6fa5 0%, #3a5a8a 100%);
            color: white;
        }
        .sidebar-title { color: white !important; }
        .sidebar-section { padding: 10px; border-radius: 5px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

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

    # Settings section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7)
    st.markdown('</div>', unsafe_allow_html=True)

    # Mode-specific controls
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
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    vector_store = process_documents({f.name: f for f in uploaded_files})
                    st.session_state.retriever = get_retriever()
                    st.session_state.pdf_processed = True
                    st.success("PDFs processed successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown('</div>', unsafe_allow_html=True)

# Main Content Area
st.markdown("<h1 style='text-align: center;'>🤖 AI Assistant</h1>", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "bot"
    if message.get("type") == "image":
        st.image(message["content"], caption='Uploaded Image', use_column_width=True)
    else:
        st.markdown(f"<div class='message {role}'>{message['content']}</div>", unsafe_allow_html=True)

# Handle different modes
if mode == "💬 Chat":
    user_input = st.chat_input("Type your message here...", key="chat_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"<div class='message user'>{user_input}</div>", unsafe_allow_html=True)

        response = get_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": response})
        st.markdown(f"<div class='message bot'>{response}</div>", unsafe_allow_html=True)

elif mode == "📄 PDF Q&A" and st.session_state.get('pdf_processed', False):
    question = st.chat_input("Ask about the PDFs...", key="pdf_question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f"<div class='message user'>{question}</div>", unsafe_allow_html=True)

        answer = ask_question(question, st.session_state.retriever)
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.markdown(f"<div class='message bot'>{answer}</div>", unsafe_allow_html=True)

elif mode == "🖼️ Image Analysis" and st.session_state.uploaded_image:
    question = st.chat_input("Ask about the image...", key="image_question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f"<div class='message user'>{question}</div>", unsafe_allow_html=True)

        response = analyze_image(st.session_state.uploaded_image, question)
        st.session_state.messages.append({"role": "bot", "content": response})
        st.markdown(f"<div class='message bot'>{response}</div>", unsafe_allow_html=True)