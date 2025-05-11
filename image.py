from groq import Groq
import base64
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def encode_image(uploaded_image, max_size=(1024, 1024)):
    """Convert uploaded image to base64 string with resizing"""
    try:
        image = Image.open(uploaded_image)
        image.thumbnail(max_size)
        buffered = BytesIO()
        image_format = image.format if image.format else 'JPEG'
        image.save(buffered, format=image_format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def analyze_image(base64_image: str, question: str) -> str:
    """Send image and question to Groq API for analysis"""
    try:
        if not base64_image:
            return "Error: No valid image provided"
            
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return f"Error analyzing image: {str(e)}"