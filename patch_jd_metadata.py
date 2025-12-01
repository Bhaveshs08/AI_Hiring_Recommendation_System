#!/usr/bin/env python3
"""
patch_jd_metadata.py
- Reads an existing JD JSON or uses the raw JD text file as source.
- If title or primary_skills missing, calls OpenAI to extract them.
- Fetches the existing vector from Pinecone (same jd_id), and upserts same vector with updated metadata.
"""
import os, sys, json, argparse
from pathlib import Path

# small helpers
def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict): return default
        cur = cur.get(k, default)
    return cur

parser = argparse.ArgumentParser()
parser.add_argument("--jd-file", required=True, help="JD JSON file path (in data/jds) or raw txt")
parser.add_argument("--jd-id", required=False, help="If provided, use this JD id in Pinecone")
parser.add_argument("--pinecone-index", required=False, help="Pinecone index name (or env PINECONE_INDEX)")
parser.add_argument("--namespace", default="Job_Descriptions")
parser.add_argument("--openai-model", default="gpt-4o-mini")
args = parser.parse_args()

PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = args.pinecone_index or os.environ.get("PINECONE_INDEX")
NAMESPACE = args.namespace

if not PINECONE_KEY or not INDEX_NAME:
    print("ERROR: PINECONE_API_KEY and PINECONE_INDEX required in environment", file=sys.stderr)
    sys.exit(2)
if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY required in environment", file=sys.stderr)
    sys.exit(3)

# import clients (same style your repo uses)
from pinecone import Pinecone
from openai import OpenAI

pc = Pinecone(api_key=PINECONE_KEY)
ix = pc.Index(INDEX_NAME)
ocl = OpenAI(api_key=OPENAI_KEY)

# load file (try JSON first, fallback to raw text)
p = Path(args.jd_file)
if not p.exists():
    print("ERROR: jd-file not found:", p, file=sys.stderr)
    sys.exit(1)

jd_raw = None
try:
    with p.open("r", encoding="utf-8-sig") as fh:
        obj = json.load(fh)
    # prefer structure: { "jd_id": "...", "metadata": {...}, ... }
    jd_raw = obj
except Exception:
    # raw text fallback
    jd_raw = {"pagecontent": p.read_text(encoding="utf-8-sig")}

# determine jd_id
jd_id = args.jd_id or jd_raw.get("jd_id") or safe_get(jd_raw, "metadata", "jd_id")
if not jd_id:
    # if file name is e.g. jd_jd_1e5c2f.json, use p.stem
    jd_id = p.stem

print("Using jd_id:", jd_id)

# get existing record from pinecone
try:
    fetched = ix.fetch(ids=[jd_id], namespace=NAMESPACE)
except Exception as e:
    print("Pinecone fetch error:", e, file=sys.stderr)
    sys.exit(1)

# normalize fetched -> get vector and meta
vecs = getattr(fetched, "vectors", None) or (fetched.get("vectors") if isinstance(fetched, dict) else {})
v_entry = None
if isinstance(vecs, dict):
    v_entry = vecs.get(jd_id)
else:
    # older SDK shape: try fetched['vectors'][jd_id]
    try:
        v_entry = fetched["vectors"][jd_id]
    except Exception:
        v_entry = None

if not v_entry:
    print("No existing JD vector found for", jd_id, "in namespace", NAMESPACE, file=sys.stderr)
    sys.exit(1)

# extract vector values and metadata (works with new/old SDK forms)
vec = None
meta = {}
if hasattr(v_entry, "values") and getattr(v_entry, "values", None):
    vec = v_entry.values
    meta = getattr(v_entry, "metadata", {}) or {}
elif isinstance(v_entry, dict):
    vec = v_entry.get("values") or v_entry.get("vector") or v_entry.get("values")
    meta = v_entry.get("metadata") or {}
else:
    # best-effort
    vec = safe_get(v_entry, "vector") or safe_get(v_entry, "values")
    meta = safe_get(v_entry, "metadata") or {}

if not vec:
    print("Warning: existing vector values not found. Cannot safely upsert with same vector.", file=sys.stderr)
    # we could generate embedding again, but avoid here
    sys.exit(1)

# find missing fields
title = meta.get("title") or jd_raw.get("title") or ""
primary_skills = meta.get("primary_skills") or jd_raw.get("primary_skills") or []

need_title = not title or str(title).strip()== ""
need_skills = not primary_skills or (isinstance(primary_skills, list) and len(primary_skills)==0)

if not (need_title or need_skills):
    print("No update needed — title and primary_skills already present.")
    sys.exit(0)

# prepare prompt for LLM to extract title and skills
text_for_extract = jd_raw.get("pagecontent") or jd_raw.get("description") or jd_raw.get("text") or ""
if not text_for_extract:
    # try reading the raw file as text
    text_for_extract = p.read_text(encoding="utf-8-sig")

prompt = f"""
You are given a job description. Extract two things as JSON only:
1) "title": a concise job title (3-6 words).
2) "primary_skills": a JSON array of important skill keywords (single words or short phrases).

Job Description:
---
{text_for_extract}
---
Return EXACTLY a JSON object, e.g.:
{{"title":"Senior Backend Engineer","primary_skills":["Java","Spring Boot","AWS","SQL","Kafka"]}}
"""

print("Calling LLM to extract title and primary_skills...")
resp = ocl.responses.create(model=args.openai_model, input=prompt)
out_text = getattr(resp, "output_text", None) or ""
if not out_text:
    # fallback deep-inspect
    if hasattr(resp, "output") and isinstance(resp.output, list) and resp.output:
        for piece in resp.output:
            if isinstance(piece, dict):
                for c in piece.get("content", []):
                    if isinstance(c, str):
                        out_text += c
                    elif isinstance(c, dict) and "text" in c:
                        out_text += c["text"]
out_text = out_text.strip()
print("LLM output:", out_text)

# try to parse JSON from LLM output
extracted = {}
try:
    extracted = json.loads(out_text)
except Exception:
    # try to find first {...} substring
    import re
    m = re.search(r"(\{.*\})", out_text, flags=re.S)
    if m:
        try:
            extracted = json.loads(m.group(1))
        except Exception:
            extracted = {}

# merge results
if need_title:
    new_title = extracted.get("title") or extracted.get("job_title") or title
    if new_title:
        meta["title"] = new_title.strip()
if need_skills:
    skills = extracted.get("primary_skills") or extracted.get("skills") or primary_skills
    # normalize to list of strings
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    meta["primary_skills"] = skills

# update source_file if missing
meta.setdefault("source_file", p.name if p.exists() else meta.get("source_file"))

# upsert back into Pinecone with same vector values
tuple_upsert = [(jd_id, vec, meta)]
try:
    ix.upsert(vectors=tuple_upsert, namespace=NAMESPACE)
    print("Successfully updated JD metadata for", jd_id)
    print("New metadata snapshot:", json.dumps(meta, indent=2, ensure_ascii=False))
except Exception as e:
    print("Pinecone upsert error:", e, file=sys.stderr)
    sys.exit(1)
