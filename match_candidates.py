# match_candidates_keywords.py
"""
Stage 1: Pinecone similarity-based matching
- Fetch JD vectors
- Match resumes
- Bucket purely by Pinecone score (semantic keyword overlap)
"""

import os
import json
import glob
from pinecone import Pinecone

# ---------- CONFIG ----------
INDEX_NAME = "prototype-index"
JD_NAMESPACE = "jd"
RESUME_NAMESPACE = "resumes"
TOP_K = 10

# Bucketing thresholds (adjustable)
HIRED_THRESHOLD = 0.75
SHORTLIST_THRESHOLD = 0.55
REJECTED_THRESHOLD = 0.30

JD_FOLDER = os.path.join(os.path.dirname(__file__), "data", "job_description")
# -----------------------------

def load_local_jds():
    files = glob.glob(os.path.join(JD_FOLDER, "*.json"))
    jds = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            j = json.load(fh)
            title = j.get("job_summary", {}).get("title", "JD")
            j["_id"] = title.lower().replace(" ", "_")
            j["_source_file"] = os.path.basename(f)
            jds.append(j)
    return jds

def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set PINECONE_API_KEY before running.")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    jds = load_local_jds()
    if not jds:
        print("[WARN] No JDs found.")
        return

    for jd in jds:
        jd_id = jd["_id"]
        title = jd.get("job_summary", {}).get("title", jd_id)

        print(f"\n=== JD: {title} ===")

        # Fetch JD vector
        jd_vec = index.fetch(ids=[jd_id], namespace=JD_NAMESPACE).vectors.get(jd_id)
        if not jd_vec:
            print(f"[WARN] JD vector {jd_id} missing in Pinecone")
            continue

        # Query Pinecone for resumes
        q = index.query(vector=jd_vec.values, top_k=TOP_K, include_metadata=True, namespace=RESUME_NAMESPACE)
        matches = q.matches

        results = []
        for m in matches:
            meta = m.metadata or {}
            name = meta.get("candidate_name_redacted") or meta.get("CandidateName") or meta.get("name", "")
            score = m.score

            # Bucketing by Pinecone similarity
            if score >= HIRED_THRESHOLD:
                bucket = "H"
            elif score >= SHORTLIST_THRESHOLD:
                bucket = "S"
            elif score <= REJECTED_THRESHOLD:
                bucket = "R"
            else:
                bucket = "N"

            results.append({
                "resume_id": m.id,
                "candidate_name": name,
                "score": round(score, 3),
                "bucket": bucket
            })

        # Print
        for r in results:
            print(f"{r['resume_id']} | {r['candidate_name']} | score={r['score']} | bucket={r['bucket']}")

if __name__ == "__main__":
    main()
