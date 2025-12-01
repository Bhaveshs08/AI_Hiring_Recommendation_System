import os
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "prototype-index"
index = pc.Index(index_name)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

folder = "data/resumes"
candidates = []

for file in os.listdir(folder):
    if file.endswith(".json"):
        path = os.path.join(folder, file)
        with open(path, "r", encoding="utf-8") as f:
            resume = json.load(f)

        # Join fields to make text for embedding
        text_parts = [
            resume.get("summary", ""),
            " ".join(resume.get("key_skills", [])),
            " ".join(resume.get("technical_skills", {}).get("languages", [])),
            " ".join(resume.get("technical_skills", {}).get("technologies", [])),
        ]
        combined_text = " ".join(text_parts)

        embedding = model.encode(combined_text).tolist()

        candidates.append({
            "id": resume["id"],
            "values": embedding,
            "metadata": {
                "name": resume.get("name", ""),
                "summary": resume.get("summary", "")
            }
        })

# Insert into Pinecone
if candidates:
    index.upsert(vectors=candidates, namespace="JD-Backend-2025")
    print(f"✅ Inserted {len(candidates)} candidates into namespace 'JD-Backend-2025'")
else:
    print("⚠️ No resumes found")
