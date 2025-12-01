# resume_to_pinecone.py
"""
Ingest JSON resumes into Pinecone with normalized metadata and a consolidated
text field for semantic search. Ensures stable IDs using candidate_id from JSON.
"""

import os, sys, json, argparse, time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# -------------------------
# Config
# -------------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")  # 3072 dims
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

if not OPENAI_KEY or not PINECONE_KEY or not PINECONE_ENV:
    print("ERROR: Set OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV in environment.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_KEY)
pc = Pinecone(api_key=PINECONE_KEY)

# -------------------------
# Helpers
# -------------------------
def load_json(path: Path):
    """Load JSON safely (strict)."""
    return json.loads(path.read_text(encoding="utf-8-sig"))

def chunk_text(text: str, chunk_size_words: int = 1500, overlap: int = 200):
    """Split text into chunks with overlap."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap
    return chunks or [""]

def get_embeddings_batch(texts, model=OPENAI_MODEL, sleep_on_error=2):
    """Batch embed texts using OpenAI, ensure valid input."""
    texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    while True:
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            print("OpenAI Embeddings error:", e)
            time.sleep(sleep_on_error)

def sanitize_metadata(md: dict):
    """Ensure Pinecone metadata contains only strings, numbers, or bools."""
    clean = {}
    for k, v in md.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, list):
            clean[k] = ", ".join(str(x) for x in v if x is not None)
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

# -------------------------
# Normalization
# -------------------------
def normalize_resume(parsed: dict, filename: str):
    """Convert raw JSON into Pinecone metadata + text."""
    candidate_id = parsed.get("candidate_id")
    if not candidate_id:
        raise ValueError(f"{filename} is missing candidate_id")

    name = parsed.get("name", "Unknown")
    email = parsed.get("email", "")
    phone = parsed.get("phone", "")
    location = parsed.get("location", "")
    current_company = parsed.get("current_company", "Unknown")
    current_role = parsed.get("current_role", "Unknown")
    exp_years = parsed.get("experience_years", 0)
    skills = parsed.get("skills", [])
    industries = parsed.get("industries", [])
    education = parsed.get("education", "")

    summary = parsed.get("summary", "")
    exp_summary = parsed.get("experience_summary", "")
    work_exp = parsed.get("experience", [])

    # Metadata (flat schema)
    metadata = {
        "candidate_id": candidate_id,
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "current_role": current_role,
        "current_company": current_company,
        "experience_years": exp_years,
        "skills": ", ".join(skills) if isinstance(skills, list) else str(skills),
        "industries": ", ".join(industries) if isinstance(industries, list) else str(industries),
        "education": education
    }
    metadata = sanitize_metadata(metadata)

    # Build consolidated text field
    document = f"""{name}
{email} | {phone}
{location}

PROFESSIONAL SUMMARY: {summary}

EXPERIENCE SUMMARY: {exp_summary}

CURRENT: {current_role} at {current_company}
Experience: {exp_years} years
Industries: {metadata['industries']}

WORK EXPERIENCE:"""
    for job in work_exp:
        role = job.get("role", "")
        company = job.get("company", "")
        document += f"\n- {role} at {company}"
        if job.get("projects"):
            document += f" ({', '.join(job.get('projects'))})"
        if job.get("responsibilities"):
            for r in job["responsibilities"]:
                document += f"\n  • {r}"

    if skills:
        document += f"\n\nSKILLS: {metadata['skills']}"
    if education:
        document += f"\n\nEDUCATION: {education}"

    return metadata, document

# -------------------------
# Pinecone Index Utils
# -------------------------
def ensure_index(index_name: str, example_vector):
    dim = len(example_vector)
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}' with dim={dim}")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
    return pc.Index(index_name)

# -------------------------
# Resume Upsert
# -------------------------
def upsert_resume_file(resume_path: Path, index, chunk_size_words=1500, overlap=200, dry_run=False, batch_upsert=100):
    parsed = load_json(resume_path)
    candidate_id = parsed.get("candidate_id")
    if not candidate_id:
        raise ValueError(f"{resume_path.name} missing candidate_id")

    metadata, document_text = normalize_resume(parsed, resume_path.name)
    chunks = chunk_text(document_text, chunk_size_words, overlap)

    vectors_for_upsert = []
    for i, chunk in enumerate(chunks):
        record_id = f"{candidate_id}_chunk{i+1}"
        md = dict(metadata)
        md.update({
            "blobType": "resume_chunk",
            "text": chunk
        })
        vectors_for_upsert.append((record_id, chunk, md))

    texts = [t for (_id, t, _md) in vectors_for_upsert]
    embeddings = get_embeddings_batch(texts)
    tuples = [(record_id, emb, md) for (record_id, _txt, md), emb in zip(vectors_for_upsert, embeddings)]

    if dry_run:
        print(f"[DRY RUN] {resume_path.name} -> candidate_id={candidate_id} chunks={len(tuples)}")
        return candidate_id, len(tuples)

    # Upsert with duplicate check
    for record_id, emb, md in tuples:
        existing = index.fetch(ids=[record_id], namespace="Resumes")
        if record_id not in existing.vectors:
            index.upsert(vectors=[(record_id, emb, md)], namespace="Resumes")

    print(f"Upserted {resume_path.name} -> candidate_id={candidate_id} chunks={len(tuples)}")
    return candidate_id, len(tuples)

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/resumes", help="Folder with JSON resume files")
    parser.add_argument("--index", default="polaris", help="Pinecone index name")
    parser.add_argument("--chunk-size", type=int, default=1500, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="overlap words")
    parser.add_argument("--batch-size", type=int, default=100, help="upsert batch size")
    parser.add_argument("--dry-run", action="store_true", help="Do everything but upsert")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = [p for p in data_dir.glob("*.json")]

    if not files:
        print("No JSON resume files found in", data_dir)
        return

    # Sample embedding for index creation
    sample_json = load_json(files[0])
    _, sample_doc = normalize_resume(sample_json, files[0].name)
    sample_emb = get_embeddings_batch([sample_doc])[0]
    idx = ensure_index(args.index, sample_emb)

    for p in tqdm(files, desc="Processing resumes"):
        try:
            upsert_resume_file(
                p, idx,
                args.chunk_size, args.overlap,
                dry_run=args.dry_run,
                batch_upsert=args.batch_size
            )
        except Exception as e:
            print("ERROR processing", p, ":", e)

if __name__ == "__main__":
    main()
