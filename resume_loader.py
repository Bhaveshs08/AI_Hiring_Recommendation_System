import os
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# === Config ===
RESUME_DIR = "data/resumes"
RESUME_NAMESPACE = "resumes"
INDEX_NAME = "prototype-index"

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(INDEX_NAME)


# === Utility: Flatten metadata ===
def flatten_metadata(resume_json):
    """Flatten nested resume fields so Pinecone accepts them as metadata."""
    flat = {}

    # Basic fields
    flat["id"] = resume_json.get("id", "")
    flat["name"] = resume_json.get("name", "")
    flat["summary"] = resume_json.get("summary", "")

    # Skills as list of strings
    skills = resume_json.get("skills", [])
    if isinstance(skills, dict):
        # Flatten all values in the dict to a single list of strings
        flat["skills"] = []
        for v in skills.values():
            if isinstance(v, list):
                flat["skills"].extend([str(item) for item in v])
            else:
                flat["skills"].append(str(v))
    elif isinstance(skills, list):
        flat["skills"] = [str(s) for s in skills]
    elif isinstance(skills, str):
        flat["skills"] = [skills]
    else:
        flat["skills"] = []


    # Experience (flatten company + designation)
    if "experience" in resume_json:
        exp_list = resume_json["experience"]
        if isinstance(exp_list, list):
            flat["experience"] = []
            for exp in exp_list:
                if isinstance(exp, dict):
                    flat["experience"].append(f"{exp.get('company', '')} - {exp.get('designation', '')}")
                elif isinstance(exp, str):
                    flat["experience"].append(exp)
        else:
            flat["experience"] = [str(exp_list)]

    return flat


# === Load and upsert resumes ===
def load_resumes():
    resume_files = [f for f in os.listdir(RESUME_DIR) if f.endswith(".json")]
    print(f"[INFO] Found {len(resume_files)} resume JSON files.")

    upserts = []

    for file in resume_files:
        path = os.path.join(RESUME_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            resume_json = json.load(f)

        # Flatten metadata for Pinecone
        metadata = flatten_metadata(resume_json)

        # Create text embedding (summary + skills + experience)
        text_parts = [
            resume_json.get("summary", ""),
            " ".join(resume_json.get("skills", [])),
        ]
        if "experience" in resume_json:
            text_parts.extend([" ".join(exp.get("responsibilities", [])) for exp in resume_json["experience"]])

        full_text = " ".join(text_parts)
        embedding = model.encode(full_text).tolist()

        # Upsert format
        upserts.append((resume_json.get("id", file), embedding, metadata))

    if upserts:
        index.upsert(vectors=upserts, namespace=RESUME_NAMESPACE)
        print(f"[INFO] Upserted {len(upserts)} resumes into namespace '{RESUME_NAMESPACE}'")
    else:
        print("[WARNING] No resumes to upsert.")


def main():
    load_resumes()


if __name__ == "__main__":
    main()
