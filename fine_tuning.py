import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import json
from db import conn

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def add_to_db(prompt, completion):
    cursor = conn.cursor()
    # Aynı kayıt varsa tekrar eklememek için kontrol
    cursor.execute("SELECT id FROM fine_tuning_data WHERE prompt=%s AND completion=%s", (prompt, completion))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO fine_tuning_data  (prompt, completion, created_at) VALUES (%s, %s, NOW())",
            (prompt, completion)
        )
        conn.commit()

def load_jsonl_to_db(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = data["messages"][1]["content"]
            completion = data["messages"][2]["content"]
            add_to_db(prompt, completion)


#openaı yükle
def upload_file(file_path):
    print("Dosya yükleniyor...")
    with open(file_path, "rb") as f: #binary okur
        response = client.files.create(file=f, purpose="fine-tune")
    return response.id


#default 3.5-turbo
def start_fine_tuning(file_id, model=None):
    load_dotenv()

    if model is None:
        model = os.getenv("BASE_OPENAI_MODEL", "gpt-3.5-turbo-0125")
    print(f"Fine-tuning başlatılıyor... (model: {model})")

    try:
        response = client.fine_tuning.jobs.create(training_file=file_id, model=model)
        print("Fine-tuning yanıtı:", response)

        # response dict ise tüm key'leri ve değerleri yazdır
        if isinstance(response, dict):
            for k, v in response.items():
                print(f"{k}: {v}")
        else:
            # Nesne ise id varsa yazdır
            print("Job ID:", getattr(response, "id", None))
        return response
    except Exception as e:
        print("Fine-tuning başlatılırken hata oluştu:", str(e))
        return None


def check_fine_tune_status(job_id):
    response = client.fine_tuning.jobs.retrieve(job_id)
    status = response.status
    print(f"Fine-tuning durumu: {status}")
    if status == "succeeded":
        print(f" Fine-tuning tamamlandı! Model adı: {response.fine_tuned_model}")
        return True
    elif status == "failed":
        print(f" Fine-tuning başarısız oldu!")
        # Hata detaylarını göster
        events = client.fine_tuning.jobs.list_events(job_id)
        for event in events.data:
            print(f"[{event.created_at}] {event.message}")
        return True
    return False

def monitor_fine_tune_job(job_id, interval=30):
    print(f"Fine-tuning job '{job_id}' izleniyor...")
    while True:
        finished = check_fine_tune_status(job_id)
        if finished:
            break
        print(f"{interval} saniye sonra tekrar kontrol edilecek...")
        time.sleep(interval)

if __name__ == "__main__":
    file_path = "fine_tune_data.jsonl"
    file_id = upload_file(file_path)
    fine_tune_response = start_fine_tuning(file_id)
    job_id = fine_tune_response.id
    check_fine_tune_status(job_id)
    monitor_fine_tune_job(job_id)


