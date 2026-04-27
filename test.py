from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")

response = client.chat.completions.create(
    model=FINE_TUNED_MODEL,
    messages=[
        {"role": "user", "content": "limitim nedir?"}
    ]
)
print(response.choices[0].message.content)


