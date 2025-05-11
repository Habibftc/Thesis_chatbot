from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
from groq import NotFoundError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Set API Key
os.environ["GROQ_API_KEY"] = "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"

# Initialize memory
memory = ConversationBufferMemory(input_key="human_input", memory_key="chat_history")

# Initialize LLM with default model
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7
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
    input_variables=["chat_history", "human_input"], template=template
)

# Initialize LLM Chain
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

def get_response(user_input):
    """Handle text conversations with error handling"""
    try:
        return llm_chain.predict(human_input=user_input)
    except NotFoundError:
        return "Error: The AI service is currently unavailable. Please try again later."
    except Exception as e:
        return f"Error processing your request: {str(e)}"