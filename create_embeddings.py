import json
from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DB_PATH = "./chroma_data"
COLLECTION_NAME = "company_data"
JSONL_FILE_PATH = "fine_tune_data_unique.jsonl"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 32  # Embedding ve ekleme için toplu işlem boyutu

print(f" Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model başarıyla yüklendi.")

print(f" ChromaDB başlatılıyor ve '{COLLECTION_NAME}' koleksiyonu yönetiliyor...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Mevcut '{COLLECTION_NAME}' koleksiyonu silindi.")
except chromadb.errors.NotFoundError:
    print(f"'{COLLECTION_NAME}' koleksiyonu bulunamadı, yeni oluşturulacak.")

collection = client.create_collection(name=COLLECTION_NAME)
print(f" Yeni veya sıfırlanmış '{COLLECTION_NAME}' koleksiyonu '{CHROMA_DB_PATH}' konumunda oluşturuldu.")

# JSONL dosyasını oku ve verileri hazırla
print(f" '{JSONL_FILE_PATH}' dosyasından veriler okunuyor...")
veriler = []
with open(JSONL_FILE_PATH, "r", encoding="utf-8") as f:
    for satir in f:
        data = json.loads(satir)
        # OpenAI sohbet formatına uygun messages listesinden assistant cevaplarını al
        for mesaj in data.get("messages", []):
            if mesaj.get("role") == "assistant":
                cevap = mesaj.get("content", "").strip()
                if cevap:  # Boş olmayan cevapları al
                    veriler.append(cevap)
print(f" {len(veriler)} adet veri başarıyla okundu.")

#  Her metni embed et ve Chroma'ya ekle
if not veriler:
    print(" Gömülecek veri bulunamadı. Lütfen JSONL dosyanızı kontrol edin.")
else:
    print(f" {len(veriler)} veri bulundu. Embedding ve kaydetme başlıyor (toplu işlem boyutu: {BATCH_SIZE})...")

    documents_to_add = []
    embeddings_to_add = []
    ids_to_add = []

    for i in range(0, len(veriler), BATCH_SIZE):
        batch_metinler = veriler[i: i + BATCH_SIZE]
        batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_metinler))]

        batch_embeddings = model.encode(batch_metinler).tolist()

        documents_to_add.extend(batch_metinler)
        embeddings_to_add.extend(batch_embeddings)
        ids_to_add.extend(batch_ids)

        print(f" {i + len(batch_metinler)} / {len(veriler)} metin embedding'leri oluşturuldu.")

    if documents_to_add:
        collection.add(
            documents=documents_to_add,
            embeddings=embeddings_to_add,
            ids=ids_to_add
        )
        print(f"Tüm {len(documents_to_add)} embedding ChromaDB'ye başarıyla eklendi.")
    else:
        print("Eklenecek doküman bulunamadı.")

print(" Embedding ve veri kaydı tamamlandı. Artık 'query_embeddings.py' dosyasını çalıştırabilirsiniz.")