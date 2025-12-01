import os
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# === CONFIGURATION ===
INDEX_NAME = "prototype-index"
JD_NAMESPACE = "jd"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pinecone API key from environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")

# === INITIALIZE ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer(EMBEDDING_MODEL)

# === LOAD JDs FROM FILES ===
JD_FOLDER = "data/job_description"
jd_files = [f for f in os.listdir(JD_FOLDER) if f.endswith(".json")]

if not jd_files:
    print("[ERROR] No JD JSON files found in folder:", JD_FOLDER)
    exit()

print(f"[INFO] Found {len(jd_files)} JD JSON files:")
for f in jd_files:
    print("  -", os.path.join(JD_FOLDER, f))

# === UPLOAD JDs TO PINECONE ===
to_upsert = []
for jd_file in jd_files:
    path = os.path.join(JD_FOLDER, jd_file)
    with open(path, "r", encoding="utf-8") as f:
        jd_json = json.load(f)

    # Assume each JD JSON has "id" and "text"
    jd_id = jd_json.get("id", jd_file.replace(".json", ""))
    jd_text = jd_json.get("text", "")
    if not jd_text:
        print(f"[WARNING] JD '{jd_file}' has empty 'text', skipping...")
        continue

    # Generate embedding
    embedding = model.encode(jd_text).tolist()

    metadata = {
        "jd_id": jd_id,
        "text": jd_text,
        "source_file": jd_file
    }

    to_upsert.append((jd_id, embedding, metadata))

# Upsert into Pinecone
if to_upsert:
    index.upsert(vectors=to_upsert, namespace=JD_NAMESPACE)
    print(f"[INFO] Upserted {len(to_upsert)} JDs into namespace '{JD_NAMESPACE}' ✅")
else:
    print("[ERROR] No JDs to upsert.")
