from endee import Endee
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = Endee("rag_db")

query = input("Enter your question: ")
query_embedding = model.encode(query).tolist()

results = db.search(vector=query_embedding, top_k=2)

print("\nRelevant Results:")
for r in results:
    print("-", r["metadata"]["text"])
