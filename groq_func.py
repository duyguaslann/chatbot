from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def get_chat_response_groq(messages, model="llama-3.1-8b-instant"):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message
