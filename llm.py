from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

# llama3-8b-8192 was decommissioned — using llama-3.1-8b-instant instead
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)










