from endee import Endee
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = Endee("rag_db")

with open("data/documents.txt", "r") as f:
    docs = f.readlines()

for i, text in enumerate(docs):
    embedding = model.encode(text).tolist()
    db.add(
        id=str(i),
        vector=embedding,
        metadata={"text": text}
    )

print("Documents stored successfully in Endee vector database")
