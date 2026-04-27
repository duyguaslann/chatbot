from sentence_transformers import SentenceTransformer
import chromadb

# --- Ayarlar ---
CHROMA_DB_PATH = "./chroma_data"
COLLECTION_NAME = "company_data"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# --- Model ve DB başlat ---
print("Model yükleniyor...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model yüklendi.")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# --- Kullanıcıdan metin al ---
raw_text = input("Lütfen eklemek istediğiniz metni girin:\n")

# --- Cümlelere ayır ve embed et ---
sentences = [s.strip() for s in raw_text.split(".") if s.strip()]
embeddings = model.encode(sentences).tolist()
ids = [f"id_{i}" for i in range(len(sentences))]

# --- ChromaDB'ye ekle ---
collection.add(
    documents=sentences,
    embeddings=embeddings,
    ids=ids
)

print("Veri başarıyla ChromaDB'ye eklendi!")
