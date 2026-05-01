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


def get_chat_response_openai(messages, tools=None, model=None):
    if model is None:
        model = os.getenv("FINE_TUNED_MODEL", "gpt-3.5-turbo")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0,
    )
    return response.choices[0].message


def openai_func(messages, user_msg, chat_id, USER_ID, model="gpt-4o", functions_to_pass=None):
    # func.json içeriğini yükle ve tools formatına çevir
    functions_path = Path(__file__).parent / "func.json"
    with open(functions_path, "r", encoding="utf-8") as f:
        functions = json.load(f)

    # Eski function formatını yeni tools formatına çevir
    tools = []
    for func in functions:
        tools.append({
            "type": "function",
            "function": func
        })

    # Chat isteği gönder
    message = get_chat_response_openai(messages, tools=tools, model=model)

    # Tool calls kontrol et (yeni API)
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name

        if function_name == "get_user_profile":
            answer = get_user_profile(int(USER_ID))
            save_message(1, user_msg, chat_id, status=1)
            save_message(0, answer, chat_id, status=1)
            return jsonify({"reply": answer})

        elif function_name == "save_note_to_desktop":
            args = json.loads(tool_call.function.arguments)
            note_text = args["note_content"]
            filename = f"not_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            desktop = Path.home() / "Desktop"
            with open(desktop / filename, "w", encoding="utf-8") as f:
                f.write(note_text)

            reply = f"Notun masaüstüne '{filename}' olarak kaydedildi."
            save_message(1, user_msg, chat_id, status=1)
            save_message(0, reply, chat_id, status=1)
            return jsonify({"reply": reply})

    # Eğer tool_call yoksa normal cevabı döndür
    reply = message.content
    save_message(1, user_msg, chat_id, status=1)
    save_message(0, reply, chat_id, status=1)
    return jsonify({"reply": reply})


def vision_chat(image_base64: str, question: str, rag_context: str = "") -> str:
    messages = []

    if rag_context:
        messages.append({
            "role": "system",
            "content": f"Aşağıdaki bağlamı kullanarak soruyu yanıtla:\n\n{rag_context}"
        })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
            {"type": "text", "text": question},
        ],
    })

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Vision API hatası: {e}")


def pdf_to_images(pdf_path: str) -> list[str]:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF kurulu değil. 'pip install pymupdf' çalıştırın.")

    import base64

    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        images.append(base64.b64encode(img_bytes).decode("utf-8"))
    doc.close()
    return images