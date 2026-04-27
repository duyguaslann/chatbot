from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
print(os.getenv("GROQ_API_KEY"))

def get_chat_response_groq(messages, model="llama3-70b-8192"):
    response = client.chat.completions.create(
        model=model,
        messages=messages
        # Groq function_call yok
    )
    return response.choices[0].message
