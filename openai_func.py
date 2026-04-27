from openai import OpenAI
import os
from dotenv import load_dotenv
from db import get_user_profile
from db import save_message
from pathlib import Path
import json
from datetime import datetime
from flask import jsonify

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")
client = OpenAI(api_key=API_KEY)


def get_chat_response_openai(messages, functions=None, model=None):
    if model is None:
        model = os.getenv("FINE_TUNED_MODEL", "gpt-3.5-turbo")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature = 0.0 ,

    )
    return response.choices[0].message


def openai_func(messages, user_msg, chat_id, USER_ID, model="gpt-4o",functions_to_pass=None):
    # func.json içeriğini yükle
    functions_path = Path(__file__).parent / "func.json"
    with open(functions_path, "r", encoding="utf-8") as f:
        functions = json.load(f)

    # Chat isteği gönder
    message = get_chat_response_openai(messages, functions=functions, model=model)

    if message.function_call:
        function_name = message.function_call.name

        if function_name == "get_user_profile":
            answer = get_user_profile(int(USER_ID))
            save_message(1, user_msg, chat_id, status=1)
            save_message(0, answer, chat_id, status=1)
            return jsonify({"reply": answer})

        elif function_name == "save_note_to_desktop":
            note_text = json.loads(message.function_call.arguments)["note_content"]
            filename = f"not_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            desktop = Path.home() / "Desktop"
            with open(desktop / filename, "w", encoding="utf-8") as f:
                f.write(note_text)

            reply = f"Notun masaüstüne '{filename}' olarak kaydedildi."
            save_message(1, user_msg, chat_id, status=1)
            save_message(0, reply, chat_id, status=1)
            return jsonify({"reply": reply})

    # Eğer function_call yoksa normal cevabı döndür
    reply = message.content
    save_message(1, user_msg, chat_id, status=1)
    save_message(0, reply, chat_id, status=1)
    return jsonify({"reply": reply})