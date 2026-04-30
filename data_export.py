"""
PostgreSQL messages tablosundan fine-tuning verisi export eder.
Cikti: training_data.jsonl
Kullanim: python data_export.py
"""
import json
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost")),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
)

OUTPUT_FILE = "training_data.jsonl"

def fetch_chat_ids(cur):
    cur.execute("""
        SELECT DISTINCT chat_id FROM messages
        WHERE status = 1
        GROUP BY chat_id
        HAVING COUNT(CASE WHEN user_type = 0 THEN 1 END) > 0
           AND COUNT(CASE WHEN user_type = 1 THEN 1 END) > 0
        ORDER BY chat_id
    """)
    return [row[0] for row in cur.fetchall()]

def fetch_messages(cur, chat_id):
    cur.execute("""
        SELECT user_type, message_text
        FROM messages
        WHERE chat_id = %s AND status = 1
        ORDER BY created_at ASC
    """, (chat_id,))
    return cur.fetchall()

def pair_messages(messages):
    pairs = []
    i = 0
    while i < len(messages) - 1:
        user_type, text = messages[i]
        next_type, next_text = messages[i + 1]
        if user_type == 0 and next_type == 1 and text and next_text:
            pairs.append({"prompt": text.strip(), "completion": next_text.strip()})
            i += 2
        else:
            i += 1
    return pairs

def main():
    with conn.cursor() as cur:
        chat_ids = fetch_chat_ids(cur)
        print(f"Toplam sohbet: {len(chat_ids)}")

        all_pairs = []
        for chat_id in chat_ids:
            messages = fetch_messages(cur, chat_id)
            pairs = pair_messages(messages)
            all_pairs.extend(pairs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Export tamamlandi: {len(all_pairs)} satir -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
