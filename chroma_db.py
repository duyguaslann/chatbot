from chromadb import PersistentClient

client = PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="chatbot_data")

def add_to_chroma(id, text, metadata={}):
    collection.add(documents=[text], ids=[id], metadatas=[metadata])

def search_chroma(query_text, n_results=3):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results
