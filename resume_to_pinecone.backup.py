import os
import json
import glob
import uuid
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm

# ========== CONFIG ==========
RESUME_DIR = "data/resumes_cleaned"
INDEX_NAME = "polaris"
NAMESPACE = "resumes"
OPENAI_MODEL = "text-embedding-3-large"  # 3072-dim embeddings
BATCH_SIZE = 100
# ============================

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def load_text(file_path):
    """Read raw resume text from .txt or .json files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            return json.load(f)
        except:
            return f.read()

def clean_metadata(value):
    """Ensure metadata is Pinecone-compatible."""
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, list):
        return [str(v) for v in value if isinstance(v, (str, int, float, bool))]
    elif isinstance(value, dict):
        # Flatten dict into a string
        return "; ".join([f"{k}: {v}" for k, v in value.items()])
    else:
        return str(value)

def normalize_resume(resume_obj, source_filename):
    # Case: resume has email and name
    if isinstance(resume_obj, dict) and "email" in resume_obj:
        name = resume_obj.get("name", "Unknown Candidate")
        email = resume_obj.get("email", "")
        # ASCII-safe ID
        id_str = f"{name.strip().replace(' ', '_')}_{email.strip()}".encode("ascii", errors="ignore").decode()
        return {
            "id": id_str,
            "email": email,
            "name": name,
            "professional_summary": resume_obj.get("professional_summary", ""),
            "experience_summary": resume_obj.get("experience_summary", ""),
            "raw_skills": resume_obj.get("raw_skills", []),
            "location": resume_obj.get("location", ""),
            "education": resume_obj.get("education", []),
            "certifications": resume_obj.get("certifications", []),
            "plaintext": resume_obj.get("plaintext", ""),
            "current_role": resume_obj.get("current_role", ""),
            "current_company": resume_obj.get("current_company", ""),
            "source_filename": source_filename
        }

    # Case: raw_text only
    elif isinstance(resume_obj, dict) and "raw_text" in resume_obj:
        text = resume_obj["raw_text"]
        return {
            "id": f"resume_{uuid.uuid4()}",
            "email": "",
            "name": "Unknown Candidate",
            "professional_summary": text[:500],
            "experience_summary": text[:800],
            "raw_skills": [],
            "location": "",
            "education": [],
            "certifications": [],
            "plaintext": text,
            "current_role": "",
            "current_company": "",
            "source_filename": source_filename
        }

    # Case: raw text fallback
    elif isinstance(resume_obj, str):
        return {
            "id": f"resume_{uuid.uuid4()}",
            "email": "",
            "name": "Unknown Candidate",
            "professional_summary": resume_obj[:500],
            "experience_summary": resume_obj[:800],
            "raw_skills": [],
            "location": "",
            "education": [],
            "certifications": [],
            "plaintext": resume_obj,
            "current_role": "",
            "current_company": "",
            "source_filename": source_filename
        }

    # Default fallback
    else:
        return {
            "id": f"resume_{uuid.uuid4()}",
            "email": "",
            "name": "Unknown Candidate",
            "professional_summary": "",
            "experience_summary": "",
            "raw_skills": [],
            "location": "",
            "education": [],
            "certifications": [],
            "plaintext": "",
            "current_role": "",
            "current_company": "",
            "source_filename": source_filename
        }

def embed_text(text):
    """Generate embedding using OpenAI."""
    response = client.embeddings.create(
        model=OPENAI_MODEL,
        input=text
    )
    return response.data[0].embedding

def process_and_upload(folder):
    files = glob.glob(os.path.join(folder, "*.txt")) + glob.glob(os.path.join(folder, "*.json"))
    batch = []

    for file_path in tqdm(files, desc="Uploading resumes"):
        try:
            raw_data = load_text(file_path)
            resume_data = normalize_resume(raw_data, os.path.basename(file_path))

            embedding_text = f"""
Professional Summary: {resume_data['professional_summary']}
Experience Summary: {resume_data['experience_summary']}
Skills: {', '.join(resume_data['raw_skills'])}
Education: {', '.join(resume_data['education'])}
Certifications: {', '.join(resume_data['certifications'])}
Current Role: {resume_data['current_role']}
Current Company: {resume_data['current_company']}
Location: {resume_data['location']}
Full Text: {resume_data['plaintext'][:2000]}
"""

            vector = embed_text(embedding_text)

            record = {
                "id": resume_data["id"],
                "values": vector,
                "metadata": {
                    "embedding_text": embedding_text,
                    "source_filename": resume_data["source_filename"],
                    "raw_skills": clean_metadata(resume_data["raw_skills"]),
                    "education": clean_metadata(resume_data["education"]),
                    "certifications": clean_metadata(resume_data["certifications"]),
                    "location": clean_metadata(resume_data["location"]),
                    "current_role": clean_metadata(resume_data["current_role"]),
                    "current_company": clean_metadata(resume_data["current_company"]),
                    "professional_summary": clean_metadata(resume_data["professional_summary"]),
                    "experience_summary": clean_metadata(resume_data["experience_summary"]),
                    "plaintext": resume_data["plaintext"][:4000]
                }
            }

            batch.append(record)

            # Upsert in batches
            if len(batch) >= BATCH_SIZE:
                index.upsert(batch, namespace=NAMESPACE)
                batch = []

        except Exception as e:
            print(f"❌ Failed {file_path}: {e}")

    # Upsert remaining records
    if batch:
        index.upsert(batch, namespace=NAMESPACE)

    print("🚀 Upload complete! Each resume = 1 record.")

if __name__ == "__main__":
    process_and_upload(RESUME_DIR)