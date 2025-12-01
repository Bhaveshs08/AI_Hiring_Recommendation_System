#!/usr/bin/env python3
"""
upload_to_pinecone.py
Ingest resume-chunk JSON files (list of chunk objects) into Pinecone.
Assumes each input JSON is a list of dicts with keys: chunk_index, section, chunk_text, metadata (candidate_id etc.)
If candidate_id is not present the script will use deterministic UUID generation as candidate seed.
Notes: requires PINECONE_API_KEY, PINECONE_INDEX, OPENAI_API_KEY (if you want notes generation)
"""
import os, sys, json, argparse, uuid, re
from pathlib import Path
from datetime import datetime
from time import sleep

from pinecone import Pinecone
from openai import OpenAI

# ensure logs dir
Path("logs").mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="Input JSON (list of chunk objects)")
parser.add_argument("--namespace", required=True, help="Pinecone namespace")
parser.add_argument("--index", required=False, help="Pinecone index name (or env)")
parser.add_argument("--id-method", choices=["uuid","email","name","content_hash"], default="uuid")
parser.add_argument("--emb-model", default="text-embedding-3-large")
parser.add_argument("--summ-model", default="gpt-4o-mini")
parser.add_argument("--batch-size", type=int, default=50)
args = parser.parse_args()

INPUT_FILE = args.file
NAMESPACE = args.namespace
INDEX_NAME = args.index or os.environ.get("PINECONE_INDEX")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ID_METHOD = args.id_method
EMB_MODEL = args.emb_model
SUMM_MODEL = args.summ_model
BATCH_SIZE = int(args.batch_size)

if not PINECONE_KEY or not INDEX_NAME:
    print("ERROR: set PINECONE_API_KEY and PINECONE_INDEX", file=sys.stderr); sys.exit(2)

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

def short_hex(s, n=6):
    import hashlib
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:n]

def slugify(text: str):
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9 _-]", " ", s)
    return "_".join(s.split())[:64] or "section"

def deterministic_candidate_id(obj: dict, method: str = "uuid"):
    meta = obj.get("metadata", {}) or {}
    name = meta.get("candidate_name") or obj.get("candidate_name") or ""
    if method == "uuid":
        return f"resumes_{uuid.uuid4().hex[:12]}"
    if method == "email":
        email = meta.get("email") or (meta.get("emails")[0] if isinstance(meta.get("emails"), (list,tuple)) and meta.get("emails") else None)
        if email:
            return f"resumes_{slugify(name or email.split('@')[0])}_{short_hex(email)}"
    if method == "name" and name:
        return f"resumes_{slugify(name)}_{short_hex(name)}"
    if method == "content_hash":
        raw = obj.get("full_text") or json.dumps(obj, sort_keys=True)
        return f"resumes_{short_hex(raw)}"
    return f"resumes_{uuid.uuid4().hex[:12]}"

def build_vector_id(candidate_id: str, chunk_index: int, section: str):
    return f"{candidate_id}_chunk{chunk_index}_{slugify(section)}"

def sanitize_meta(meta: dict):
    out = {}
    for k,v in (meta or {}).items():
        if isinstance(v, str):
            v2 = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", v).strip()
            out[k] = v2
        elif isinstance(v, list):
            out[k] = [ (re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", x).strip() if isinstance(x,str) else x) for x in v ]
        else:
            out[k] = v
    return out

# load file with utf-8-sig
if not Path(INPUT_FILE).exists():
    print("ERROR: Input file not found:", INPUT_FILE, file=sys.stderr); sys.exit(1)

with open(INPUT_FILE, "r", encoding="utf-8-sig") as fh:
    raw = json.load(fh)

if not isinstance(raw, list):
    print("ERROR: input JSON must be a list of chunk objects", file=sys.stderr); sys.exit(1)

first = raw[0] if raw else {}
seed_obj = {"metadata": {"candidate_name": first.get("candidate_name") or (first.get("metadata") or {}).get("candidate_name"), "email": (first.get("metadata") or {}).get("email")}, "full_text": first.get("full_text") or json.dumps(raw, sort_keys=True)}
candidate_id = deterministic_candidate_id(seed_obj, ID_METHOD)
print(f"Candidate ID: {candidate_id}  (method={ID_METHOD})")

# normalize
normalized = []
for i,obj in enumerate(raw, start=1):
    meta = (obj.get("metadata") or {}).copy()
    meta.setdefault("candidate_name", meta.get("candidate_name") or first.get("candidate_name",""))
    meta.setdefault("email", meta.get("email") or (first.get("metadata") or {}).get("email",""))
    section = obj.get("section") or meta.get("section") or f"section{i}"
    chunk_index = obj.get("chunk_index") or i
    vid = build_vector_id(candidate_id, chunk_index, section)
    text = (obj.get("chunk_text") or obj.get("text") or obj.get("full_text") or "").strip()
    if not text:
        print("Skipping empty chunk:", vid)
        continue
    meta["candidate_id"] = candidate_id
    meta["section"] = section
    meta["source_file"] = meta.get("source_file") or Path(INPUT_FILE).name
    meta["version"] = meta.get("version", 1)
    meta["updated_at"] = datetime.now().isoformat()
    normalized.append({"id": vid, "text": text, "metadata": meta})

if not normalized:
    print("No chunks to ingest after normalization.")
    sys.exit(0)

print(f"Prepared {len(normalized)} chunks for {candidate_id}")

def make_prompt(section, text):
    sec = (section or "SECTION").upper().replace(" ", "_")
    return f"You are a resume summarizer. Produce a short one-line NOTE summarizing this section.\nSECTION: {sec}\n\nTEXT:\n{text}\n\nOutput: {sec}: <one-line summary>"

def gen_notes(items):
    out = []
    if not client:
        # fallback: create snippet only
        for it in items:
            text = it["text"].replace("\n", " ")
            snippet = " ".join(text.split())[:240]
            sec_label = (it["metadata"].get("section","SECTION")).upper().replace(" ", "_")
            out.append(f"{sec_label}: {snippet}")
        return out

    for it in items:
        prompt = make_prompt(it["metadata"].get("section"), it["text"])
        try:
            resp = client.responses.create(model=SUMM_MODEL, input=prompt)
            note_text = getattr(resp, "output_text", None) or ""
            if not note_text and hasattr(resp, "output") and isinstance(resp.output, list) and resp.output:
                piece = resp.output[0]
                if isinstance(piece, dict):
                    for c in piece.get("content", []):
                        if isinstance(c, str):
                            note_text += c
                        elif isinstance(c, dict) and "text" in c:
                            note_text += c["text"]
            note_text = note_text.strip()
        except Exception as e:
            print("Notes error:", e, file=sys.stderr)
            note_text = ""
        # ensure starts with SECTION_NAME:
        sec_label = (it["metadata"].get("section","SECTION")).upper().replace(" ", "_")
        if not note_text:
            text = it["text"].replace("\n", " ")
            snippet = " ".join(text.split())[:240]
            note_text = f"{sec_label}: {snippet}"
        elif not note_text.upper().startswith(sec_label + ":"):
            note_text = sec_label + ": " + note_text
        out.append(note_text)
    return out

# embed helper
def embed_texts(texts):
    try:
        emb_resp = client.embeddings.create(model=EMB_MODEL, input=texts) if client else None
        if emb_resp:
            return [r.embedding for r in emb_resp.data]
        else:
            # placeholder zeros if OpenAI not enabled (avoid upsert of None)
            return [[0.0]*1536 for _ in texts]  # note: dimension depends on model; replace as needed
    except Exception as e:
        print("Embedding failed:", e, file=sys.stderr)
        raise

# batch process
i = 0
total = len(normalized)
while i < total:
    batch = normalized[i:i+BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE+1}: items {i+1}-{i+len(batch)} / {total}")
    notes = gen_notes(batch)
    for it,n in zip(batch, notes):
        it["metadata"]["notes"] = n
    texts = [b["text"] for b in batch]
    embeddings = embed_texts(texts)
    if len(embeddings) != len(batch):
        print("Embedding length mismatch", file=sys.stderr); sys.exit(1)
    upsert_tuples = []
    for it,vec in zip(batch, embeddings):
        upsert_tuples.append((it["id"], vec, sanitize_meta(it["metadata"])))
    try:
        index.upsert(vectors=upsert_tuples, namespace=NAMESPACE)
        print("Upserted", len(upsert_tuples), "vectors.")
    except Exception as e:
        print("Pinecone upsert failed:", e, file=sys.stderr)
        sys.exit(1)
    i += BATCH_SIZE
    sleep(0.15)

print("SUCCESS: uploaded", len(normalized), "chunks for", candidate_id)
