from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from groq import NotFoundError
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your API key (use environment variables in production)
os.environ["GROQ_API_KEY"] = "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"

# Initialize memory with updated parameters
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="human_input",
    return_messages=True
)

# Initialize LLM with the default model
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7,
)

# Define Prompt
template = """
The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

# Initialize LLM Chain with updated configuration
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def get_response(user_input):
    """Handle text conversations with error handling"""
    try:
        return llm_chain.invoke({"human_input": user_input})["text"]
    except NotFoundError:
        return "Error: The AI service is currently unavailable. Please try again later."
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return f"Error processing your request: {str(e)}"