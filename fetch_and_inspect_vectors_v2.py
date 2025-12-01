# fetch_and_inspect_vectors_v2.py
import os, json
from pinecone import Pinecone

api = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX")
if not api or not index_name:
    raise SystemExit("Set PINECONE_API_KEY and PINECONE_INDEX in environment.")

pc = Pinecone(api_key=api)
idx = pc.Index(index_name)

resumes_ns = "Resumes"
jds_ns = "Job_Descriptions"

def flatten_listing(lst):
    ids = []
    for item in lst:
        if isinstance(item, list):
            # flatten nested lists recursively
            ids.extend(flatten_listing(item))
        elif isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            if item.get("id"):
                ids.append(item.get("id"))
            elif item.get("vector_id"):
                ids.append(item.get("vector_id"))
        else:
            if hasattr(item, "id"):
                ids.append(getattr(item, "id"))
    return ids

print("Calling index.list() for Resumes and JDs (limit=100)...")
res_listing = list(idx.list(namespace=resumes_ns, limit=100))
jd_listing = list(idx.list(namespace=jds_ns, limit=100))

print("\\nRaw resume listing repr (truncated):")
print(repr(res_listing)[:3000])
print("\\nRaw JD listing repr (truncated):")
print(repr(jd_listing)[:3000])

res_ids = flatten_listing(res_listing)
jd_ids = flatten_listing(jd_listing)

print("\\nFlattened Resume IDs found:", res_ids)
print("Flattened JD IDs found:", jd_ids)

def fetch_and_report(ids, namespace, label):
    if not ids:
        print(f"No ids to fetch for {label}")
        return
    print(f"\\nFetching {len(ids)} ids from namespace '{namespace}' ...")
    fetched = idx.fetch(ids=ids, namespace=namespace)
    # convert to dict if possible
    try:
        d = fetched.to_dict()
    except Exception:
        try:
            d = dict(fetched)
        except Exception:
            print("Could not convert fetched object to dict; repr:")
            print(repr(fetched)[:4000])
            return
    vectors = d.get("vectors") or {}
    if isinstance(vectors, dict) and vectors:
        for i, _id in enumerate(ids, start=1):
            ent = vectors.get(_id)
            if ent is None:
                print(f"{i}. {_id} -> FETCHED: None")
                continue
            keys = list(ent.keys())
            has_values = ("values" in ent) or ("vector" in ent) or ("values" in ent.get("vector", {}))
            print(f"{i}. {_id} -> keys: {keys} -> has_values: {has_values}")
            vec = None
            if "values" in ent:
                vec = ent["values"]
            elif "vector" in ent:
                v = ent["vector"]
                if isinstance(v, dict) and "values" in v:
                    vec = v["values"]
                elif isinstance(v, list):
                    vec = v
            if vec:
                print(f"    vector length: {len(vec)} (preview first 8): {vec[:8]}")
    else:
        print("Fetched object did not contain a 'vectors' dict or it was empty; full fetched repr (truncated):")
        print(json.dumps(d, indent=2)[:4000])

fetch_and_report(res_ids, resumes_ns, "Resumes")
fetch_and_report(jd_ids, jds_ns, "Job_Descriptions")
