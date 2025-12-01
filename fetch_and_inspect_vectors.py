# fetch_and_inspect_vectors.py
import os, json
from pinecone import Pinecone

api = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX")
if not api or not index_name:
    raise SystemExit("Set PINECONE_API_KEY and PINECONE_INDEX in environment.")

pc = Pinecone(api_key=api)
idx = pc.Index(index_name)

# Namespaces observed
resumes_ns = "Resumes"
jds_ns = "Job_Descriptions"

# get listing (strings or dicts)
res_listing = list(idx.list(namespace=resumes_ns, limit=100))
jd_listing = list(idx.list(namespace=jds_ns, limit=100))

print("Raw resume listing repr:")
print(repr(res_listing)[:4000])
print("\nRaw JD listing repr:")
print(repr(jd_listing)[:4000])

# normalize to string ids
def ids_from_listing(lst):
    ids = []
    for item in lst:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            if item.get("id"):
                ids.append(item.get("id"))
            elif item.get("vector_id"):
                ids.append(item.get("vector_id"))
        else:
            # object-like
            if hasattr(item, "id"):
                ids.append(getattr(item, "id"))
    return ids

res_ids = ids_from_listing(res_listing)
jd_ids = ids_from_listing(jd_listing)

print("\nResume IDs found:", res_ids)
print("JD IDs found:", jd_ids)

def fetch_and_report(ids, namespace, label):
    if not ids:
        print(f"No ids to fetch for {label}")
        return
    print(f"\nFetching {len(ids)} ids from namespace '{namespace}' ...")
    fetched = idx.fetch(ids=ids, namespace=namespace)
    # Try to convert to dict
    try:
        d = fetched.to_dict()
    except Exception:
        try:
            d = dict(fetched)
        except Exception:
            print("Could not convert fetched object to dict; repr:")
            print(repr(fetched)[:4000])
            return
    # expected shape: {'vectors': {id: {...}}}
    vectors = d.get("vectors") or {}
    if isinstance(vectors, dict):
        for i, _id in enumerate(ids, start=1):
            ent = vectors.get(_id)
            if ent is None:
                print(f"{i}. {_id} -> FETCHED: None")
                continue
            keys = list(ent.keys())
            has_values = ("values" in ent) or ("vector" in ent) or ("values" in ent.get("vector", {}))
            print(f"{i}. {_id} -> keys: {keys} -> has_values: {has_values}")
            # If values present, print length of vector
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
        print("Fetched object did not contain a 'vectors' dict; full fetched repr:")
        print(json.dumps(d, indent=2)[:4000])

fetch_and_report(res_ids, resumes_ns, "Resumes")
fetch_and_report(jd_ids, jds_ns, "Job_Descriptions")
