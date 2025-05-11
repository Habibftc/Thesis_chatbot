from groq import Groq
from groq import APITimeoutError, APIError
import base64
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def encode_image(uploaded_image):
    """Convert uploaded image to base64 string with error handling"""
    try:
        image = Image.open(uploaded_image)
        buffered = BytesIO()
        image_format = image.format if image.format else 'JPEG'
        image.save(buffered, format=image_format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        raise Exception("Failed to process the image. Please check the file and try again.")

def analyze_image(base64_image, question):
    """Send image and question to Groq API for analysis with retry mechanism"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client = Groq(
                api_key=os.getenv("GROQ_API_KEY"),
                timeout=30  # Set timeout for the client
            )

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
                timeout=25  # Set timeout for the request
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
            if attempt == max_retries - 1:  # Only return error on last attempt
                return f"Error analyzing image: {str(e)}"
    
    return "Sorry, I couldn't analyze the image after multiple attempts."