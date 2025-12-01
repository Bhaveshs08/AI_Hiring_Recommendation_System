# match_candidates_llm.py
import os
import json
import re
import sys
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INDEX_NAME = "prototype-index"
JD_NAMESPACE = "jd"
RESUME_NAMESPACE = "resumes"
OPENAI_MODEL = "gpt-4o-mini"

# ---------------- ENV CHECK ----------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    print("Set PINECONE_API_KEY environment variable and re-run.")
    sys.exit(1)
if not OPENAI_API_KEY:
    print("Set OPENAI_API_KEY environment variable and re-run.")
    sys.exit(1)

# ---------------- CLIENTS ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
llm = OpenAI(api_key=OPENAI_API_KEY)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = embed_model.get_sentence_embedding_dimension()  # should match your index dimension

# ---------------- HELPERS ----------------
def extract_matches(result):
    """Return a list of match dict-like objects from various Pinecone return shapes."""
    if result is None:
        return []
    # Pinecone Python objects sometimes provide .matches, sometimes dict with "matches"
    if hasattr(result, "matches"):
        return result.matches
    if isinstance(result, dict):
        return result.get("matches", [])
    # fallback
    return []

def match_to_plain(m):
    """Normalize a match (object/dict) to a plain dict with id, score, metadata, values (if present)."""
    if hasattr(m, "id"):
        _id = getattr(m, "id", None)
        _score = getattr(m, "score", None)
        _metadata = getattr(m, "metadata", {}) or {}
        _values = getattr(m, "values", None) or getattr(m, "vector", None)
    elif isinstance(m, dict):
        _id = m.get("id")
        _score = m.get("score")
        _metadata = m.get("metadata", {}) or {}
        _values = m.get("values") or m.get("vector") or m.get("values")
    else:
        _id = None
        _score = None
        _metadata = {}
        _values = None
    return {"id": _id, "score": _score, "metadata": _metadata, "values": _values}

def clean_gpt_response(raw_output: str) -> str:
    """Strip markdown code fences and surrounding prose to extract JSON text."""
    txt = raw_output.strip()
    # Remove triple backticks blocks (```json ... ``` or ``` ... ```)
    if txt.startswith("```"):
        # remove first ```... and trailing ```
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
        txt = re.sub(r"\n?```$", "", txt)
        return txt.strip()
    # If the model responded with prose containing a JSON block, extract it
    # naive approach: find first '{' or '[' and last corresponding '}' or ']'
    first_brace = min(
        [pos for pos in (txt.find('['), txt.find('{')) if pos != -1],
        default=-1
    )
    if first_brace == -1:
        return txt
    # Try to extract JSON substring by finding last '}' or ']'
    last_brace = max(txt.rfind('}'), txt.rfind(']'))
    if last_brace == -1:
        return txt
    return txt[first_brace:last_brace + 1]

def safe_json_load(s: str):
    """Attempt to parse JSON from string, return object or raise."""
    cleaned = clean_gpt_response(s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to salvage by searching for JSON substring
        m = re.search(r'(\{.*\}|\[.*\])', s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise

# ---------------- PINECONE UTILS ----------------
def describe_stats():
    stats = index.describe_index_stats()
    return stats

def list_namespace_count(namespace):
    stats = describe_stats()
    return stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)

def fetch_jds():
    """Return list of JD matches: each item is dict {'id','text','metadata'}."""
    count = list_namespace_count(JD_NAMESPACE)
    if count == 0:
        return []
    # query using a zero vector but ask for up to `count` results
    res = index.query(vector=[0.0] * EMBED_DIM, top_k=count, include_metadata=True, namespace=JD_NAMESPACE)
    matches = extract_matches(res)
    items = []
    for m in matches:
        md = match_to_plain(m)["metadata"]
        # prefer explicit 'text' field, else construct one
        if md.get("text"):
            text = md.get("text")
        else:
            title = md.get("title", "")
            exp = md.get("experience_required", "")
            tech = ", ".join(md.get("technical_skills", [])) if isinstance(md.get("technical_skills", []), list) else md.get("technical_skills", "")
            primary = ", ".join(md.get("primary_skills", [])) if isinstance(md.get("primary_skills", []), list) else md.get("primary_skills", "")
            secondary = ", ".join(md.get("secondary_skills", [])) if isinstance(md.get("secondary_skills", []), list) else md.get("secondary_skills", "")
            text = f"{title} role requiring {exp}. Technical skills: {tech}. Primary: {primary}. Secondary: {secondary}."
        items.append({"id": match_to_plain(m)["id"], "text": text, "metadata": md})
    return items

def query_resumes_by_jd_embedding(jd_embedding):
    """Query resume namespace using JD embedding; returns normalized match list."""
    count = list_namespace_count(RESUME_NAMESPACE)
    if count == 0:
        return []
    res = index.query(vector=jd_embedding, top_k=count, include_metadata=True, namespace=RESUME_NAMESPACE)
    matches = extract_matches(res)
    norm = [match_to_plain(m) for m in matches]
    return norm

# ---------------- LLM PROMPT ----------------
def build_prompt(jd_text, candidates):
    """
    candidates: list of dicts: {resume_id, name, pine_score, metadata...}
    Returns text prompt for the LLM.
    """
    short_cands = []
    for c in candidates:
        short_cands.append({
            "resume_id": c.get("resume_id"),
            "name": c.get("name"),
            "pinecone_score": c.get("pinecone_score"),
            "skills": c.get("metadata", {}).get("raw_skills", []),
            "job_titles": c.get("metadata", {}).get("job_titles", [])
        })
    prompt = f"""You are an expert technical recruiter. Evaluate each candidate against the Job Description (JD) below.

Job Description:
{jd_text}

Candidates (with Pinecone similarity scores):
{json.dumps(short_cands, ensure_ascii=False, indent=2)}

For each candidate return a JSON object with:
- resume_id: string (must match the id provided)
- name: candidate name (or 'Unknown')
- gpt_score: float between 0.0 and 1.0 (higher = better)
- bucket: one of "H" (Hired), "S" (Shortlisted), "R" (Rejected), "N" (Non-domain)
- reason: short (1-3 sentence) explanation for the decision.

Return ONLY a JSON array (no extra text). Example output:
[
  {{
    "resume_id": "resume-1",
    "name": "Alice",
    "gpt_score": 0.87,
    "bucket": "H",
    "reason": "Matches required Java + Spark experience and has relevant projects."
  }},
  ...
]
"""
    return prompt

# ---------------- MAIN WORKFLOW ----------------
def main():
    print("=== match_candidates_llm.py starting ===")

    # 1) Check index stats
    stats = describe_stats()
    print("[INFO] index stats summary:", {"dimension": stats.get("dimension"), "namespaces": list(stats.get("namespaces", {}).keys())})
    # Confirm dimension match
    idx_dim = stats.get("dimension")
    if idx_dim and idx_dim != EMBED_DIM:
        print(f"[WARN] index dimension ({idx_dim}) != embedding model dimension ({EMBED_DIM}). Proceed carefully.")

    # 2) Load JDs
    jds = fetch_jds()
    if not jds:
        print("[ERROR] No job descriptions found in Pinecone namespace 'jd'. Run job_description_loader.py first.")
        return
    # choose first JD for now
    jd = jds[0]
    jd_text = jd["text"]
    print(f"[INFO] Using JD id={jd['id']} (len={len(jd_text)} chars)")

    # 3) Create JD embedding (ensure same model used when upserting JDs if possible)
    jd_embedding = embed_model.encode(jd_text).tolist()

    # 4) Query resumes by JD embedding
    resume_matches = query_resumes_by_jd_embedding(jd_embedding)
    if not resume_matches:
        print("[ERROR] No resumes found in Pinecone namespace 'resumes'. Run resume loader first.")
        return
    print(f"[INFO] Found {len(resume_matches)} resume matches from Pinecone")

    # 5) Prepare candidate list for LLM
    candidates = []
    for m in resume_matches:
        rid = m.get("id")
        md = m.get("metadata", {}) or {}
        # possible fields for name: candidate_name_redacted, CandidateName, name
        name = md.get("candidate_name_redacted") or md.get("CandidateName") or md.get("name") or md.get("candidate_name") or "Unknown"
        candidates.append({
            "resume_id": rid,
            "name": name,
            "pinecone_score": m.get("score", 0.0),
            "metadata": md
        })

    # 6) Build prompt and call GPT
    prompt = build_prompt(jd_text, candidates)
    print("[INFO] Calling GPT-4o-mini for re-ranking (this may take a few seconds)...")
    response = llm.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "You are a strict JSON-only responder."},
                  {"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = response.choices[0].message.content
    try:
        parsed = safe_json_load(raw)
    except Exception as e:
        print("[ERROR] Could not parse GPT output as JSON.")
        print("Raw GPT output:")
        print(raw)
        raise

    # parsed should be a list (JSON array)
    if isinstance(parsed, dict) and parsed.get("evaluations"):
        parsed_list = parsed.get("evaluations")
    elif isinstance(parsed, list):
        parsed_list = parsed
    elif isinstance(parsed, dict):
        # if LLM returned object mapping, try to extract array-like values
        parsed_list = parsed.get("results") or parsed.get("evaluations") or []
    else:
        parsed_list = []

    # 7) Print final results in requested format
    print("\nFinal candidate evaluations:")
    for item in parsed_list:
        rid = item.get("resume_id", "unknown")
        name = item.get("name", "Unknown")
        gpt_score = item.get("gpt_score", "N/A")
        bucket = item.get("bucket", "?")
        reason = item.get("reason", "")
        print(f"{rid} | name:{name} | gpt_score:{gpt_score} | bucket:{bucket} | reason: {reason}")

    # 8) Save to output file
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"{jd.get('id','jd')}_llm_re_ranked.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed_list, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved re-ranked output to: {out_path}")

if __name__ == "__main__":
    main()
