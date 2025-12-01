import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
INDEX_NAME = "prototype-index"
JD_NAMESPACE = "jd"

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example JD
jd_id = "jd-1"
jd_text = "We are looking for a Python developer with experience in web frameworks and REST APIs."

# Create embedding
jd_embedding = model.encode(jd_text).tolist()

# Upsert JD into Pinecone
index.upsert(
    vectors=[(jd_id, jd_embedding, {"jd": jd_text})],
    namespace=JD_NAMESPACE
)

print(f"[INFO] Upserted JD '{jd_id}' into namespace '{JD_NAMESPACE}'")