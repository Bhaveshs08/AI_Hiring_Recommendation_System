#!/usr/bin/env python3
"""
match_all_resumes_vs_jds.py

Compare all resume vectors in Pinecone namespace "Resumes" against all JD vectors in
namespace "Job_Descriptions". Output rows: Resume ID | JD ID | Score (cosine similarity).

Requirements:
- Pinecone v4 client installed (`pinecone-client`)
- PINECONE_API_KEY and PINECONE_INDEX set in environment (activate_env.ps1)
"""

import os
import sys
import json
import csv
import math
from typing import List, Dict, Any

# try numpy for speed/accuracy; fall back to pure-Python
try:
    import numpy as np
except Exception:
    np = None

try:
    from pinecone import Pinecone
except Exception as e:
    raise SystemExit("Missing dependency: pinecone (v4). Install with: python -m pip install 'pinecone-client'") from e

# ---------- Helpers ----------
def cosine_sim(a: List[float], b: List[float]) -> float:
    if a is None or b is None:
        return -1.0
    if np is not None:
        a_np = np.array(a, dtype=float)
        b_np = np.array(b, dtype=float)
        denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np))
        if denom == 0:
            return -1.0
        return float(np.dot(a_np, b_np) / denom)
    # pure python fallback
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)

def safe_extract_vector(entry: Any) -> List[float]:
    """
    Accepts dict/obj shapes that Pinecone might return and attempts to extract
    a vector/list of floats from common keys.
    """
    if entry is None:
        return None
    # dict-like
    if isinstance(entry, dict):
        # common keys: 'values', 'vector', 'values' nested under 'vector' etc.
        for k in ("values", "vector", "values_list", "values_vector"):
            if k in entry and entry[k] is not None:
                v = entry[k]
                if isinstance(v, dict) and "values" in v:
                    v = v["values"]
                return [float(x) for x in v] if v is not None else None
        # sometimes the metadata contains embedding; avoid using metadata here
        return None
    # object-like
    for attr in ("values", "vector"):
        if hasattr(entry, attr):
            v = getattr(entry, attr)
            if v is None:
                continue
            if isinstance(v, dict) and "values" in v:
                v = v["values"]
            return [float(x) for x in v]
    # nothing found
    return None

# ---------- Pinecone wrapper ----------
class PineconeHelper:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        try:
            self.index = self.pc.Index(index_name)
        except Exception as e:
            raise RuntimeError(f"Could not access index '{index_name}': {e}")

    def list_vectors(self, namespace: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return list of vector entries in the namespace. Handles generator/list/dict shapes.
        Each item in returned list is either a dict (with possible keys 'id','metadata','values'...) or a converted dict.
        """
        gen = self.index.list(namespace=namespace, limit=limit)
        out = []
        if gen is None:
            return out
        if isinstance(gen, dict):
            # common shape: {'vectors': [...]}
            if "vectors" in gen and isinstance(gen["vectors"], list):
                return gen["vectors"][:limit]
            # flatten any list value
            for k, v in gen.items():
                if isinstance(v, list):
                    return v[:limit]
            return [gen]
        # if generator/iterable
        if hasattr(gen, "__iter__") and not isinstance(gen, str):
            try:
                for item in gen:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        out.append(item)
                    else:
                        # try to coerce object to dict with common attrs
                        d = {}
                        if hasattr(item, "id"):
                            d["id"] = getattr(item, "id")
                        if hasattr(item, "metadata"):
                            d["metadata"] = getattr(item, "metadata")
                        # values/ vector
                        if hasattr(item, "values"):
                            d["values"] = getattr(item, "values")
                        elif hasattr(item, "vector"):
                            d["vector"] = getattr(item, "vector")
                        out.append(d)
                return out
            except TypeError:
                pass
        # fallback
        return [gen]

    def fetch_vectors(self, ids: List[str], namespace: str) -> Dict[str, Any]:
        """
        Fetch by ids and return dict keyed by id containing vector/metadata
        """
        fetched = self.index.fetch(ids=ids, namespace=namespace)
        # fetched may be object with .to_dict() or dict-like
        if hasattr(fetched, "to_dict"):
            fetched = fetched.to_dict()
        # expected shapes:
        # {'vectors': {id: { 'metadata':..., 'values': [...] }, ...}}
        vectors = {}
        if isinstance(fetched, dict):
            if "vectors" in fetched and isinstance(fetched["vectors"], dict):
                vectors = fetched["vectors"]
            elif "vectors" in fetched and isinstance(fetched["vectors"], list):
                # list of dicts
                for v in fetched["vectors"]:
                    vid = v.get("id") or v.get("vector_id")
                    if vid:
                        vectors[vid] = v
            else:
                # try keys for matches/items
                for k in ("matches", "items", "results"):
                    if k in fetched and isinstance(fetched[k], list):
                        for v in fetched[k]:
                            vid = v.get("id") or v.get("vector_id")
                            if vid:
                                vectors[vid] = v
        else:
            # object-like fallback not implemented further
            pass
        return vectors

# ---------- Main flow ----------
def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX")
    if not api_key or not index_name:
        print("ERROR: Set PINECONE_API_KEY and PINECONE_INDEX in environment (activate_env.ps1).")
        sys.exit(2)

    helper = PineconeHelper(api_key=api_key, index_name=index_name)

    resumes_ns = "Resumes"
    jds_ns = "Job_Descriptions"

    print("Listing resume vectors in namespace:", resumes_ns)
    resumes = helper.list_vectors(namespace=resumes_ns, limit=100)
    print("Found resumes:", len(resumes))

    print("Listing JD vectors in namespace:", jds_ns)
    jds = helper.list_vectors(namespace=jds_ns, limit=100)
    print("Found JDs:", len(jds))

    if len(resumes) == 0:
        print("No resume vectors found in namespace:", resumes_ns)
        sys.exit(0)
    if len(jds) == 0:
        print("No JD vectors found in namespace:", jds_ns)
        sys.exit(0)

    # Build id -> vector maps for both sets. If vector not present in listing, try fetch.
    resume_map = {}
    resume_ids_missing_vec = []
    for r in resumes:
        rid = r.get("id") or r.get("vector_id")
        if not rid:
            continue
        vec = None
        # try inline values
        if "values" in r:
            vec = r.get("values")
        elif "vector" in r:
            vec = r.get("vector")
        vec = safe_extract_vector(r) if vec is None else (vec if isinstance(vec, list) else vec.get("values") if isinstance(vec, dict) else None)
        if vec is None:
            resume_ids_missing_vec.append(rid)
        resume_map[rid] = {"vector": vec, "metadata": r.get("metadata") or r.get("meta") or {}}

    jd_map = {}
    jd_ids_missing_vec = []
    for j in jds:
        jid = j.get("id") or j.get("vector_id")
        if not jid:
            continue
        vec = None
        if "values" in j:
            vec = j.get("values")
        elif "vector" in j:
            vec = j.get("vector")
        vec = safe_extract_vector(j) if vec is None else (vec if isinstance(vec, list) else vec.get("values") if isinstance(vec, dict) else None)
        if vec is None:
            jd_ids_missing_vec.append(jid)
        jd_map[jid] = {"vector": vec, "metadata": j.get("metadata") or j.get("meta") or {}}

    # If any vectors missing, fetch them by id
    if resume_ids_missing_vec:
        print(f"Fetching missing resume vectors ({len(resume_ids_missing_vec)}) via fetch()...")
        fetched = helper.fetch_vectors(resume_ids_missing_vec, namespace=resumes_ns)
        for rid in resume_ids_missing_vec:
            entry = fetched.get(rid)
            vec = safe_extract_vector(entry) if entry else None
            resume_map[rid]["vector"] = vec

    if jd_ids_missing_vec:
        print(f"Fetching missing JD vectors ({len(jd_ids_missing_vec)}) via fetch()...")
        fetched = helper.fetch_vectors(jd_ids_missing_vec, namespace=jds_ns)
        for jid in jd_ids_missing_vec:
            entry = fetched.get(jid)
            vec = safe_extract_vector(entry) if entry else None
            jd_map[jid]["vector"] = vec

    # Confirm we have vectors
    resume_vec_count = sum(1 for v in resume_map.values() if v["vector"])
    jd_vec_count = sum(1 for v in jd_map.values() if v["vector"])
    print(f"Resume vectors available: {resume_vec_count}/{len(resume_map)}")
    print(f"JD vectors available: {jd_vec_count}/{len(jd_map)}")

    if resume_vec_count == 0 or jd_vec_count == 0:
        print("ERROR: Missing vectors. Cannot compute scores.")
        sys.exit(1)

    # Compute pairwise scores and write CSV
    out_rows = []
    for rid, rinfo in resume_map.items():
        rvec = rinfo["vector"]
        if not rvec:
            continue
        for jid, jinfo in jd_map.items():
            jvec = jinfo["vector"]
            if not jvec:
                continue
            score = cosine_sim(rvec, jvec)
            out_rows.append((rid, jid, score))

    # Sort by descending score if desired
    out_rows_sorted = sorted(out_rows, key=lambda x: x[2], reverse=True)

    # Print table header
    print("\nResume ID | JD ID | Score")
    for rid, jid, score in out_rows_sorted:
        print(f"{rid} | {jid} | {score:.6f}")

    # Save CSV
    csv_file = "resume_jd_scores.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["resume_id", "jd_id", "score"])
        for rid, jid, score in out_rows_sorted:
            writer.writerow([rid, jid, f"{score:.6f}"])

    print(f"\nSaved {len(out_rows_sorted)} rows to {csv_file}")

if __name__ == "__main__":
    main()
