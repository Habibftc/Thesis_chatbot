from groq import Groq
import base64
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Set API Key
os.environ["GROQ_API_KEY"] = "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"


def encode_image(uploaded_image):
    """Convert uploaded image to base64 string"""
    image = Image.open(uploaded_image)
    buffered = BytesIO()
    image_format = image.format if image.format else 'JPEG'
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_image(base64_image, question):
    """Send image and question to Groq API for analysis"""
    try:
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
        return f"Error analyzing image: {str(e)}"