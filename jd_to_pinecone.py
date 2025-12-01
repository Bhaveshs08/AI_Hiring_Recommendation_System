#!/usr/bin/env python3
"""
jd_to_pinecone.py

Load job-description JSON files from data/jds (or convert from jds_raw),
generate a short embedding-text field, and upsert each JD into Pinecone under namespace Job_Descriptions.

Usage:
  python jd_to_pinecone.py --data-dir data/jds
"""
import os, sys, json, argparse, re
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/jds")
parser.add_argument("--namespace", default="Job_Descriptions")
parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX"))
parser.add_argument("--emb-model", default="text-embedding-3-large")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
NAMESPACE = args.namespace
INDEX_NAME = args.index
EMB_MODEL = args.emb_model
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_KEY or not INDEX_NAME:
    print("ERROR: set PINECONE_API_KEY and PINECONE_INDEX", file=sys.stderr); sys.exit(1)
pc = Pinecone(api_key=PINECONE_KEY)
ix = pc.Index(INDEX_NAME)
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

def build_text_for_embedding(j):
    # combine title, skills, description, experience into one text for embedding
    parts = []
    if j.get("title"):
        parts.append(j["title"])
    meta = j.get("metadata") or {}
    if meta.get("primary_skills"):
        if isinstance(meta["primary_skills"], list):
            parts.append(", ".join(meta["primary_skills"]))
        else:
            parts.append(str(meta["primary_skills"]))
    if j.get("description"):
        parts.append(j["description"][:25000])
    if meta.get("experience_required"):
        parts.append(str(meta["experience_required"]))
    return "\n".join(parts)

for f in sorted(DATA_DIR.glob("*.json")):
    with f.open("r", encoding="utf-8-sig") as fh:
        j = json.load(fh)
    # ensure metadata keys
    j.setdefault("metadata", {})
    j["metadata"].setdefault("source_file", f.name)
    j.setdefault("title", j.get("title") or f.stem)
    jd_id = j.get("jd_id") or j.get("metadata", {}).get("jd_id") or f.stem
    emb_text = build_text_for_embedding(j)
    # generate embedding
    if not client:
        print("ERROR: OPENAI_API_KEY required to generate embeddings", file=sys.stderr); sys.exit(1)
    emb_resp = client.embeddings.create(model=EMB_MODEL, input=[emb_text])
    vec = emb_resp.data[0].embedding
    # prepare metadata to upsert
    meta = j.get("metadata", {})
    meta.update({"title": j.get("title",""), "experience_required": j.get("experience_required",""), "primary_skills": meta.get("primary_skills",[])})
    try:
        ix.upsert(vectors=[(jd_id, vec, meta)], namespace=NAMESPACE)
        print(f"Upserted {f.name} -> jd_id={jd_id}")
    except Exception as e:
        print("Upsert failed:", e, file=sys.stderr)
