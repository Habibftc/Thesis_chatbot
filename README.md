# Large Model Driven Digital Human Q&A System 🤖

This project is a multi-functional chatbot system powered by large language models. It supports text-based conversations, PDF document-based Q&A, and image analysis. The system is built using Python and Streamlit for an interactive user interface.

## Features
- **💬 Chat Mode**: Engage in friendly conversations with the AI.
- **📄 PDF Q&A Mode**: Upload PDF documents and ask questions based on their content.
- **🖼️ Image Analysis Mode**: Upload images and ask questions about them.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd thesis-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your API key:
   ```properties
   GROQ_API_KEY="your_api_key_here"
   ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the application in your browser at `http://localhost:8501`.

3. Use the sidebar to select a mode:
   - **Chat**: Type your message and interact with the AI.
   - **PDF Q&A**: Upload PDF files, process them, and ask questions.
   - **Image Analysis**: Upload an image and ask questions about it.

## Project Structure
- `app.py`: Main application file for the Streamlit interface.
- `model.py`: Handles text-based conversations.
- `pdf_model.py`: Processes PDF documents and retrieves relevant information.
- `image.py`: Encodes and analyzes images using the Groq API.
- `.env`: Stores environment variables like the API key.

## Requirements
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) for the interactive UI.
- [LangChain](https://langchain.com/) for document processing and embeddings.
- [Groq API](https://groq.com/) for image and text analysis.

Feel free to contribute to this project by submitting issues or pull requests!
