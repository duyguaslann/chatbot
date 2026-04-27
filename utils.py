import json
from db import conn

def export_to_jsonl(output_path="fine_tune_data.jsonl"):
    with conn.cursor() as cur:
        cur.execute("SELECT prompt, completion FROM fine_tuning_data")
        rows = cur.fetchall()

    with open(output_path, "w", encoding="utf-8") as f:
        for prompt, completion in rows:
            json_line = {
                "prompt": prompt.strip() + "\n",
                "completion": completion.strip() + "\n"
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"JSONL dosyası oluşturuldu: {output_path}")

if __name__ == "__main__":
    export_to_jsonl()
