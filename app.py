from flask import Flask, render_template, request, jsonify, redirect, session, url_for
import os
from dotenv import load_dotenv
from db import (
    create_chat, get_chats, save_message, get_messages_by_chat, update_chat_title, clear_chat_history,
    get_user_fullname, get_user_profile, get_user_by_email, verify_password, conn, get_last_messages
)
from datetime import datetime
import json
from common_file import is_time_query, is_currency_query, foreign_currency
from pathlib import Path
from openai_func import get_chat_response_openai, openai_func, vision_chat, pdf_to_images
from groq_func import get_chat_response_groq
from ollama_func import get_chat_response_ollama
from db import get_today_message_count
from openai import OpenAI
from fine_tuning import load_jsonl_to_db, upload_file
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import pandas as pd

load_dotenv()

print("FINE_TUNED_MODEL:", os.getenv("FINE_TUNED_MODEL"))
USE_PROVIDER = os.getenv("USE_PROVIDER", "groq").lower()

if USE_PROVIDER == "openai":
    LLM_MODEL = "gpt-4o"
elif USE_PROVIDER == "groq":
    LLM_MODEL = "llama-3.1-8b-instant"
elif USE_PROVIDER == "ollama":
    LLM_MODEL = "llama3.2"
else:
    raise ValueError("USE_PROVIDER değeri 'openai', 'groq' ya da 'ollama' olmalı.")

# Model ve ChromaDB ayarları
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
collection = chroma_client.get_or_create_collection(name="company_data")

SIMILARITY_SCORE = float(os.getenv("SIMILARITY_THRESHOLD", "0.28"))
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", 3))
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "gpt-3.5-turbo")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "gizli-anahtar")  # session kullanabilmek için oluştur

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception("Unhandled exception")
    return jsonify({"error": str(e)}), 500

if USE_PROVIDER == "openai":
    FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "gpt-3.5-turbo")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif USE_PROVIDER == "groq":
    client = None  # Groq için fine-tuning yok
else:
    raise ValueError("USE_PROVIDER değeri 'groq' ya da 'openai' olmalı.")

# Benzer içerik arama
def search_similar_contexts(user_prompt):
    embedding = model.encode([user_prompt])[0]  #Kullanıcının mesajını embedding'e çeviriyor
    results = collection.query(query_embeddings=[embedding], n_results=MAX_CONTEXT)  #benzer MAX_CONTEXT kadar dökümanı çekiyor
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Benzerlik eşiğine göre filtrele
    filtered = [(doc, dist) for doc, dist in zip(documents, distances) if dist <= SIMILARITY_SCORE]

    # Terminalde benzerlik skorlarını yazdır
    print("Benzerlik skorları ve dokümanlar:")
    for doc, dist in zip(documents, distances):
        print(f"Score: {dist:.4f} - Doc: {doc[:80]}...")

    context_texts = [doc for doc, dist in filtered]
    return context_texts


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    if not email or not password:
        return render_template("homepage.html", error="Email ve şifre gerekli")

    user = get_user_by_email(email)
    if user and verify_password(password, user["password"]):
        # Başarılı girişte session'a kaydet
        session["user_id"] = user["id"]
        session["user_name"] = f"{user['first_name']} {user['last_name']}"
        return redirect(url_for("redirect_page"))
    else:
        return render_template("homepage.html", error="Kullanıcı bulunamadı ya da şifre yanlış")

# Provider switch
@app.route('/set-provider', methods=['POST'])
def set_provider():
    data = request.get_json()
    provider = data.get('provider')
    if provider in ['openai', 'groq', 'ollama']:
        session['provider'] = provider
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "Invalid provider"}), 400

#Chat mesajı işleme fonksiyonu (placeholder)
def process_message(message):
    provider = session.get('provider', 'openai')  # varsayılan
    if provider == 'openai':
        pass
    else:
        #Groq işlemleri
        pass

#Chatbot sayfası
@app.route("/chatler")
def chatler():
    user_id = session.get("user_id")
    if not user_id:
        #Giriş yapılmamışsa anasayfaya yönlendir
        return redirect("/")

    chats = get_chats(user_id)
    user_name = session.get("user_name", "Kullanıcı")
    return render_template("chatler.html", chats=chats, user_name=user_name)

# Yeni chat oluştur
@app.route("/chat/create", methods=["POST"])
def create_new_chat():
    chat = create_chat(session["user_id"])
    if not chat:
        return jsonify({"detail": "Chat oluşturulamadı"}), 400
    return jsonify({
        "chat_id": chat["id"],
        "user_id": chat["user_id"],
        "title": chat["title"]
    })

@app.route("/chats")
def list_chats():
    chats = get_chats(session["user_id"])
    return jsonify(chats)

@app.route("/history/<int:chat_id>")
def chat_history(chat_id):
    messages = get_messages_by_chat(chat_id)
    return jsonify({"history": messages})


# Günlük mesaj limiti
@app.route("/message-limit", methods=["GET"])
def get_message_limit():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Giriş yapılmamış"}), 401

    limit = int(os.getenv("DAILY_MESSAGE_LIMIT", 20))
    count = get_today_message_count(user_id)
    return jsonify({"used": count, "limit": limit})

#Chat mesaj gönder ve sil
@app.route("/chat/<int:chat_id>", methods=["POST", "DELETE"])
def chat_message(chat_id):
    try:
        if request.method == "DELETE":
            from db import delete_chat
            delete_chat(chat_id)
            return jsonify({"message": "Chat silindi"})

        # POST method
        data = request.get_json()
        if data is None:
            return jsonify({"detail": "Geçersiz JSON verisi"}), 400

        user_msg = data.get("message")
        if not user_msg:
            return jsonify({"detail": "Mesaj gereklidir"}), 400

        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"detail": "Kullanıcı girişi gerekli."}), 401

        # Günlük mesaj limiti kontrolü
        DAILY_LIMIT = int(os.getenv("DAILY_MESSAGE_LIMIT", 20))
        current_count = get_today_message_count(user_id)
        if current_count >= DAILY_LIMIT:
            return jsonify({"reply": "Bugünlük mesaj limitinize ulaştınız. Lütfen yarın tekrar deneyin."}), 200

        user_msg_lower = user_msg.lower()

        # Saat ve tarih sorgusu
        if is_time_query(user_msg_lower):
            now = datetime.now()
            tarih_str = now.strftime("Tarih %d %B %Y %A. Saat %H:%M")
            save_message(1, user_msg, chat_id, status=1)
            save_message(0, tarih_str, chat_id, status=1)
            return jsonify({"reply": tarih_str})

        # Döviz sorgusu
        if is_currency_query(user_msg_lower):
            reply = foreign_currency(user_msg_lower)
            save_message(1, user_msg, chat_id, status=1)
            save_message(0, reply, chat_id, status=1)
            return jsonify({"reply": reply})

        # Benzer contextleri ara
        similar_contexts = search_similar_contexts(user_msg)
        context_text = "\n\n".join(similar_contexts)

        # json file (openai)
        functions_path = Path(__file__).parent / "func.json"
        with open(functions_path, "r", encoding="utf-8") as f:
            functions = json.load(f)

        # Eğer context varsa, prompt'a ekle
        if context_text:
            full_prompt = f"{user_msg}\n\nBu bilgilere göre cevap ver:\n{context_text}"
        else:
            full_prompt = user_msg

        system_message = (
            "Sen CMA Danışmanlık hakkında bilgi veren yardımcı asistansın. Sorulara net ve doğru cevaplar ver."
            " Kullanıcı not tutmak isterse, 'save_note_to_desktop' fonksiyonunu çağır."
            " Kullanıcı ben kimim derse, 'get_user_profile' fonksiyonunu çağır."
        )

        # OpenAI API'ye çağrı yap
        messages_to_send = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ]

        provider = session.get('provider', USE_PROVIDER)

        if provider == 'openai':
            response = openai_func(messages_to_send, user_msg, chat_id, session["user_id"], model=FINE_TUNED_MODEL, functions_to_pass=functions)
            print("Kullanılan model:", FINE_TUNED_MODEL)
            if response:
                return response

        elif provider == 'groq':
            LLM_MODEL = "llama-3.1-8b-instant"
            message = get_chat_response_groq(messages_to_send, model=LLM_MODEL)
            if message:
                save_message(1, user_msg, chat_id, status=1)
                save_message(0, message.content, chat_id, status=1)
                return jsonify({"reply": message.content})

        elif provider == 'ollama':
            LLM_MODEL = "llama3.2"
            message = get_chat_response_ollama(messages_to_send, model=LLM_MODEL)
            if message:
                save_message(1, user_msg, chat_id, status=1)
                save_message(0, message.content, chat_id, status=1)
                return jsonify({"reply": message.content})
        else:
            return jsonify({"error": "Geçersiz provider seçimi"}), 400

    except Exception as e:
        app.logger.exception("Chat mesajı gönderilirken hata oluştu")
        return jsonify({"error": str(e)}), 500

@app.route("/chat/<int:chat_id>/title", methods=["PUT"])
def update_chat_title_api(chat_id):
    data = request.get_json()
    new_title = data.get("title")
    if not new_title:
        return jsonify({"detail": "Yeni başlık gerekli"}), 400
    update_chat_title(chat_id, new_title)
    return jsonify({"detail": "Başlık güncellendi"})

@app.route("/chat/<int:chat_id>/clear_history", methods=["POST"])
def clear_history(chat_id):
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE messages SET status = 0 WHERE chat_id = %s", (chat_id,))
            conn.commit()
        return jsonify({"message": "Chat geçmişi temizlendi"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/training")
def training_page():
    return render_template("training.html")

@app.route('/redirect.html')
def redirect_page():
    return render_template('redirect.html')

@app.route("/query")
def query_page():
    return render_template('query.html')

@app.route("/training/data", methods=["GET"])
def get_training_data():
    with conn.cursor() as cur:
        cur.execute("SELECT id, prompt, completion, created_at FROM fine_tuning_data ORDER BY created_at DESC")
        rows = cur.fetchall()
        data = [
            {
                "id": r[0],
                "prompt": r[1],  # Kullanıcı sorusu
                "completion": r[2],  # Asistan cevabı
                "created_at": r[3].strftime("%Y-%m-%d %H:%M")
            }
            for r in rows
        ]
        return jsonify(data)



@app.route('/api/models', methods=['GET'])
def get_all_models():
    with conn.cursor() as cur:
        cur.execute("SELECT id, model_name FROM models ORDER BY created_at DESC")
        rows = cur.fetchall()
        return jsonify([{'id': r[0], 'name': r[1]} for r in rows])

# Fine tuning başlat
@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    model_row_id = data.get('base_model_id')

    if not model_row_id:
        return jsonify({"error": "Model ID gerekli"}), 400

    try:
        with conn.cursor() as cur:
            # id'ye karşılık model_name'i al
            cur.execute("SELECT model_name FROM models WHERE id = %s", (model_row_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Model bulunamadı"}), 404
            model_name = row[0]

            training_file_id = upload_file("fine_tune_data.jsonl")

            response = client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model_name  # string model adı
            )

            job_id = response.id
            print(f"Model adı: {model_name}")
            print(f"Yüklenen dosya ID'si: {training_file_id}")
            print(f"Fine-tune job ID: {response.id}")
            return jsonify({"status": "başlatıldı", "job_id": response.id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def export_finetuning_data_to_jsonl():
    system_msg = (
        "Sen CMA Danışmanlık hakkında bilgi veren yardımcı asistansın. Sorulara net ve doğru cevaplar ver."
        " Kullanıcı not tutmak isterse, 'save_note_to_desktop' fonksiyonunu çağır."
        " Kullanıcı ben kimim derse, 'get_user_profile' fonksiyonunu çağır."
    )

    jsonl_path = "fine_tune_data.jsonl"
    # Eski dosyayı sil
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        print("Eski dosya silindi.")

    # Veritabanından tüm veriyi çek ve yeni dosyayı oluştur
    with conn.cursor() as cur:
        cur.execute("SELECT prompt, completion FROM fine_tuning_data ORDER BY created_at ASC")
        rows = cur.fetchall()

        print(f"Export edilen kayıt sayısı: {len(rows)}")

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for prompt, completion in rows:
                obj = {
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt.strip()},
                        {"role": "assistant", "content": completion.strip()}
                    ]
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print("Yeni JSONL dosyası başarıyla oluşturuldu.")

@app.route("/fine-tune/status", methods=["GET"])
def fine_tune_status():
    if USE_PROVIDER != "openai":
        return jsonify({"error": "Fine-tuning sadece OpenAI sağlayıcısında destekleniyor."}), 400

    job_id = request.args.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id parametresi gerekli"}), 400

    try:
        response = client.fine_tuning.jobs.get(id=job_id)
        return jsonify({"status": response.status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/load-jsonl", methods=["POST"])
def load_jsonl():
    try:
        with conn.cursor() as cur:
            #Önce tablodaki tüm verileri sil
            cur.execute("DELETE FROM fine_tuning_data")
            conn.commit()

        load_jsonl_to_db("fine_tune_data.jsonl")
        return jsonify({"status": "success", "message": "JSONL dosyası veritabanına yüklendi ve eski veriler silindi"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

#Fine-tuning durum sorgulama
@app.route('/api/train/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    try:
        response = client.fine_tuning.jobs.retrieve(job_id)
        status = response.status
        return jsonify({"status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/training/data", methods=["POST"])
def add_training_data():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    completion = data.get("completion", "").strip()

    if not prompt or not completion:
        return jsonify({"error": "Prompt ve completion zorunludur."}), 400

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO fine_tuning_data (prompt, completion) VALUES (%s, %s)",
            (prompt, completion)
        )
        conn.commit()

    export_finetuning_data_to_jsonl()  # Her eklemeden sonra JSONL dosyasını güncelle

    return jsonify({"message": "Kayıt başarıyla eklendi."})

@app.route("/training/data/<int:data_id>", methods=["DELETE"])
def delete_training_data(data_id):
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM fine_tuning_data WHERE id = %s", (data_id,))
            conn.commit()
        export_finetuning_data_to_jsonl()
        return jsonify({"message": "Kayıt silindi."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["GET"])
def query():
    return render_template("query.html")

@app.route("/add-data", methods=["POST"])
def add_data():
    data = request.get_json()
    raw_text = data.get("text", "").strip()

    if not raw_text:
        return {"error": "Boş metin gönderilemez."}, 400

    sentences = [s.strip() for s in raw_text.split(".") if s.strip()]
    embeddings = model.encode(sentences).tolist()
    ids = [f"id_{uuid.uuid4()}" for _ in sentences]

    collection.add(documents=sentences, embeddings=embeddings, ids=ids)
    return {"message": "Veriler başarıyla eklendi!"}

@app.route("/veriler", methods=["GET"])
def verileri_getir():
    results = collection.get()  # ChromaDB'den veri al
    data = [
        {"id": id_, "text": doc}
        for id_, doc in zip(results["ids"], results["documents"])
    ]
    return jsonify(data)

@app.route("/delete-data", methods=["POST"])
def delete_data():
    data = request.get_json()
    ids = data.get("ids", [])

    if not ids:
        return {"error": "Silinecek veri seçilmedi."}, 400

    collection.delete(ids=ids)
    return {"message": "Seçilen veriler silindi."}

@app.route("/update-data", methods=["POST"])
def update_data():
    data = request.get_json()
    doc_id = data.get("id")
    new_text = data.get("text", "").strip()

    if not doc_id or not new_text:
        return {"error": "ID ve yeni metin zorunludur."}, 400

    # Önce sil, güncellenmiş veriyi tekrar ekle
    collection.delete(ids=[doc_id])

    new_embedding = model.encode([new_text]).tolist()
    collection.add(documents=[new_text], embeddings=new_embedding, ids=[doc_id])

    return {"message": "Veri güncellendi."}


@app.route("/upload-excel", methods=["POST"])
def upload_excel():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Dosya gönderilmedi."}), 400

    try:
        df = pd.read_excel(file, engine='openpyxl')

        if "Soru" not in df.columns or "Cevap" not in df.columns:
            return jsonify({"error": "Excel sütun adları 'Soru' ve 'Cevap' olmalı."}), 400

        cur = conn.cursor()

        for _, row in df.iterrows():
            prompt = row["Soru"].strip()
            completion = row["Cevap"].strip()

            if not completion.endswith("\n"):
                completion += "\n"

            cur.execute(
                "INSERT INTO fine_tuning_data (prompt, completion) VALUES (%s, %s)",
                (prompt, completion)
            )
        conn.commit()
        cur.close()

        export_finetuning_data_to_jsonl()

        return jsonify({"message": "Veriler başarıyla fine_tuning_data tablosuna eklendi ve jsonl oluşturuldu."}), 200

    except Exception as e:
       return jsonify({"error": str(e)}), 500



ALLOWED_VISION_EXTENSIONS = {"jpg", "jpeg", "png", "pdf"}

@app.route("/api/vision", methods=["POST"])
def vision_endpoint():
    file = request.files.get("file")
    question = request.form.get("question", "Bu görseli açıkla.")

    if not file:
        return jsonify({"error": "Dosya gönderilmedi."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_VISION_EXTENSIONS:
        return jsonify({"error": f"Desteklenmeyen dosya tipi: {ext}. İzin verilenler: {', '.join(ALLOWED_VISION_EXTENSIONS)}"}), 400

    rag_context = ""
    try:
        query_embedding = model.encode([question]).tolist()
        rag_results = collection.query(query_embeddings=query_embedding, n_results=3)
        if rag_results["documents"] and rag_results["documents"][0]:
            rag_context = "\n".join(rag_results["documents"][0])
    except Exception:
        pass

    try:
        if ext == "pdf":
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            try:
                pages = pdf_to_images(tmp_path)
            finally:
                os.unlink(tmp_path)

            answers = []
            for i, page_b64 in enumerate(pages):
                answer = vision_chat(page_b64, question, rag_context)
                answers.append({"page": i + 1, "answer": answer})
            return jsonify({"results": answers})

        else:
            import base64
            image_b64 = base64.b64encode(file.read()).decode("utf-8")
            answer = vision_chat(image_b64, question, rag_context)
            return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)

































