from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DB_PATH = "./chroma_data"
COLLECTION_NAME = "company_data"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

print("Model yükleniyor...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model yüklendi.")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


all_data = collection.get(include=["documents"])
tum_metinler = [doc for sublist in all_data['documents'] for doc in sublist]

tum_metinler = all_data['documents']
print("ChromaDB'deki tüm metinler:")
for i, metin in enumerate(tum_metinler, 1):
    print(f"{i}. {metin}")

while True:
    metin = input("Yeni metin gir (çıkmak için q): ").strip()
    if metin.lower() == 'q':
        print("Program sonlandırıldı.")
        break
    if metin == "":
        print("Boş metin girdiniz, lütfen tekrar deneyin.")
        continue

    print(f"Girilen metin: '{metin}'")  # Kontrol için

    embedding = model.encode([metin]).tolist()
    new_id = f"user_doc_{collection.count()}"
    collection.add(documents=[metin], embeddings=embedding, ids=[new_id])
    print(f"Metin başarıyla eklendi. Toplam kayıt sayısı: {collection.count()}\n")

    # Kullanıcıdan metinle ilgili soru al
    soru = input("Bu metinle ilgili sorunuzu yazın (devam etmek için Enter, çıkmak için 'q'): ").strip()
    if soru.lower() == 'q':
        print("Program sonlandırıldı.")
        break
    if soru == "":
        print("Soru boş, yeni metin eklemeye devam edebilirsiniz.\n")
        continue

    soru_embedding = model.encode([soru]).tolist()
    sonuclar = collection.query(
        query_embeddings=soru_embedding,
        n_results=3,
        include=["documents", "distances"]
    )

    print("--- Sorgu Sonuçları ---")
    if sonuclar['documents'] and sonuclar['documents'][0]:
        for i, (doc, dist) in enumerate(zip(sonuclar['documents'][0], sonuclar['distances'][0])):
            print(f"{i+1}. Benzerlik (Uzaklık): {dist:.4f}")
            print(f"   Metin: {doc}\n")
    else:
        print("Eşleşen sonuç bulunamadı.")


