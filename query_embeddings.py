from sentence_transformers import SentenceTransformer
import chromadb


CHROMA_DB_PATH = "./chroma_data"
COLLECTION_NAME = "company_data"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

print(f" Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}...")
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Model başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Model yüklenirken hata oluştu: {e}")
    exit()

print(f"🔁 ChromaDB başlatılıyor ve '{COLLECTION_NAME}' koleksiyonu yükleniyor...")
try:
    # PersistentClient kullanarak diskteki veritabanına bağlan
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"✅ '{COLLECTION_NAME}' koleksiyonu '{CHROMA_DB_PATH}' konumundan yüklendi.")
    print(f"Koleksiyondaki toplam doküman sayısı: {collection.count()}")

except chromadb.errors.NotFoundError:
    print(f"❌ Hata: '{COLLECTION_NAME}' koleksiyonu '{CHROMA_DB_PATH}' konumunda bulunamadı.")
    print("Lütfen önce embedding'leri oluşturmak ve kaydetmek için 'create_embeddings.py' dosyasını çalıştırın.")
    exit()
except Exception as e:
    print(f"❌ ChromaDB başlatılırken veya koleksiyon yüklenirken hata oluştu: {e}")
    exit()

soru = "CMA ortakları kimlerdir"

print(f"\n❓ Sorgulanıyor: '{soru}'")
try:
    soru_embedding = model.encode([soru])[0].tolist()

    sonuclar = collection.query(
        query_embeddings=[soru_embedding],
        n_results=3,
        include=["documents", "distances"]
    )

    print("--- Sorgu Sonuçları ---")
    if sonuclar['documents'] and sonuclar['documents'][0]:
        for i, (dokuman_metni, uzaklik) in enumerate(zip(sonuclar['documents'][0], sonuclar['distances'][0])):
            print(f"{i+1}. Benzerlik (Uzaklık): {uzaklik:.4f}")
            print(f"   Metin: {dokuman_metni[:150]}...") 
    else:
        print("Eşleşen sonuç bulunamadı.")

except Exception as e:
    print(f"❌ Sorgulama sırasında hata oluştu: {e}")

print("---")
print("✅ Sorgu işlemi tamamlandı.")




































