import json

input_file = "fine_tune_data.jsonl"
output_file = "fine_tune_data_unique.jsonl"

unique_assistant_responses = set()
unique_data_entries = []

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)

        # Sadece assistant rolündeki contente bak
        assistant_content = ""
        for message in data.get("messages", []):
            if message.get("role") == "assistant":
                assistant_content = message.get("content", "").strip()
                break

        if assistant_content:
            if assistant_content not in unique_assistant_responses:
                unique_assistant_responses.add(assistant_content)
                unique_data_entries.append(data)  # JSON objesinin tamamını ekle

print(f"'{input_file}' dosyasında {len(unique_assistant_responses)} adet benzersiz assistant cevabı bulundu.")
print(f"{len(unique_data_entries)} adet benzersiz JSON satırı oluşturuldu.")

#  yeni bir jsonl dosyasına yaz
with open(output_file, "w", encoding="utf-8") as outfile:
    for entry in unique_data_entries:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Benzersiz veriler '{output_file}' dosyasına yazıldı.")