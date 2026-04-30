"""
dokuman/Chatbot .docx dosyasini okuyup ChromaDB'ye yukler.
Kullanim: python load_chroma.py
"""
import json
import requests
import docx

DOCX_PATH = r"dokuman\Chatbot .docx"
ADD_DATA_URL = "http://localhost:5000/add-data"

# SQL ve env satırları gibi teknik/kod parçalarını atla
SKIP_PREFIXES = (
    "CREATE ", "id integer", "title ", "created_at ", "status integer",
    "user_id ", "CONSTRAINT ", ");", "pip install", "import ",
    "from ", "USE_PROVIDER=", "GROQ_API_KEY=", "OPENAI_API_KEY=",
    "DB_HOST=", "DB_NAME=", "DB_USER=", "DB_PASS=", "DAILY_MESSAGE_LIMIT=",
    "BASE_OPENAI_MODEL=", "FINE_TUNED_MODEL=", "USER_HISTORY_SIMILARITY_SCORE=",
    "MAX_USER_HISTORY_CONTEXT=", "client =", "collection =", "model =",
    "prompt text", "completion text", "message_text", "chat_id", "user_type",
    "email varchar", "first_name", "last_name", "password text",
    "note text,", "age integer,",
)

def is_useful(text):
    t = text.strip()
    if len(t) < 15:
        return False
    for prefix in SKIP_PREFIXES:
        if t.startswith(prefix):
            return False
    return True

def load_docx_paragraphs(path):
    doc = docx.Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

def chunk_paragraphs(paragraphs, size=5):
    useful = [p for p in paragraphs if is_useful(p)]
    chunks = []
    for i in range(0, len(useful), size):
        chunk = " ".join(useful[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def send_chunk(text):
    resp = requests.post(ADD_DATA_URL, json={"text": text}, timeout=30)
    return resp.status_code, resp.json()

if __name__ == "__main__":
    print("Doküman okunuyor...")
    paragraphs = load_docx_paragraphs(DOCX_PATH)
    print(f"Toplam paragraf: {len(paragraphs)}")

    chunks = chunk_paragraphs(paragraphs, size=5)
    print(f"Gönderilecek chunk sayısı: {len(chunks)}\n")

    success = 0
    for i, chunk in enumerate(chunks):
        status, resp = send_chunk(chunk)
        if status == 200:
            success += 1
            print(f"[{i+1}/{len(chunks)}] OK")
        else:
            print(f"[{i+1}/{len(chunks)}] HATA {status}: {resp}")

    print(f"\nTamamlandı: {success}/{len(chunks)} chunk yüklendi.")
