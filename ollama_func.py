import ollama
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
_ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

def get_chat_response_ollama(messages, model="llama3.2"):
    try:
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ("system", "user", "assistant")
        ]
        response = _ollama_client.chat(model=model, messages=ollama_messages)
        return response['message']
    except Exception as e:
        print(f"Ollama error: {e}")
        return None