#!/usr/bin/env python3
"""
unify_resumes.py

Usage:
  python unify_resumes.py --from-prefix <old_prefix> --to-id <canonical_candidate_id> [--delete-old]

Example:
  python unify_resumes.py --from-prefix sumeet_adhav_001 --to-id resumes_sumeet_adhav_5a07e6 --delete-old

Behavior:
 - Lists all vector ids in namespace 'Resumes' starting with the given from-prefix.
 - Fetches vector values + metadata for those ids.
 - Constructs new vector ids by replacing the prefix part up to '_chunk' with the given to-id
   (preserves chunk index + section suffix).
 - Updates metadata['candidate_id'] to to-id.
 - Upserts the new vectors into the same namespace.
 - If --delete-old is provided, deletes the original vectors after successful upsert.
"""
import os, sys, argparse, time
from pinecone import Pinecone

parser = argparse.ArgumentParser()
parser.add_argument("--from-prefix", required=True, help="Prefix of old vectors (e.g. sumeet_adhav_001)")
parser.add_argument("--to-id", required=True, help="Canonical candidate_id to unify under (e.g. resumes_sumeet_adhav_5a07e6)")
parser.add_argument("--namespace", default="Resumes", help="Pinecone namespace")
parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX"), help="Pinecone index name (env PINECONE_INDEX if not provided)")
parser.add_argument("--delete-old", action="store_true", help="Delete old vectors after re-upsert")
parser.add_argument("--batch-size", type=int, default=100, help="Fetch/upsert batch size")
args = parser.parse_args()

API_KEY = os.environ.get("PINECONE_API_KEY")
if not API_KEY or not args.index:
    print("ERROR: PINECONE_API_KEY and index name required (env PINECONE_INDEX or --index)."); sys.exit(2)

pc = Pinecone(api_key=API_KEY)
ix = pc.Index(args.index)
ns = args.namespace
from_pref = args.from_prefix
to_id = args.to_id

print(f"Scanning namespace '{ns}' for ids starting with '{from_pref}' ...")
# collect ids (use list pagination)
found_ids = []
try:
    for page in ix.list(namespace=ns, limit=100):
        # normalize page
        if isinstance(page, str):
            found_ids.append(page)
        elif isinstance(page, list):
            for p in page:
                if isinstance(p, str):
                    found_ids.append(p)
                elif isinstance(p, dict) and "id" in p:
                    found_ids.append(p["id"])
        elif isinstance(page, dict):
            # older SDK shapes
            for k in page.keys():
                found_ids.append(k)
except Exception as e:
    print("List failed:", e)
    sys.exit(1)

found_ids = [i for i in found_ids if isinstance(i, str) and i.startswith(from_pref)]
found_ids = list(dict.fromkeys(found_ids))
print(f"Found {len(found_ids)} vector ids to process.")

if not found_ids:
    print("No vectors found for that prefix. Exiting.")
    sys.exit(0)

# process in batches
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

all_new_ids = []
for batch in chunks(found_ids, args.batch_size):
    print(f"Fetching batch of {len(batch)}")
    fetched = ix.fetch(ids=batch, namespace=ns)
    vecs = getattr(fetched, "vectors", None) or (fetched.get("vectors") if isinstance(fetched, dict) else {})
    to_upsert = []
    for old_id in batch:
        vobj = vecs.get(old_id) if isinstance(vecs, dict) else None
        if vobj is None:
            print("  WARNING: missing vector for", old_id)
            continue
        # vector values
        values = getattr(vobj, "values", None) or (vobj.get("values") if isinstance(vobj, dict) else None)
        if values is None:
            print("  WARNING: no values for", old_id); continue
        # metadata
        meta = getattr(vobj, "metadata", None) or (vobj.get("metadata") if isinstance(vobj, dict) else {}) or {}
        # derive suffix part after prefix: find first occurrence of '_chunk'
        if "_chunk" in old_id:
            suffix = old_id.split("_chunk",1)[1]
            new_id = f"{to_id}_chunk{suffix}"
        else:
            # fallback: replace prefix string at start
            new_id = old_id.replace(from_pref, to_id, 1)
        # ensure metadata candidate_id replaced
        meta["candidate_id"] = to_id
        to_upsert.append((new_id, list(values), meta))
        all_new_ids.append(new_id)
    if not to_upsert:
        continue
    try:
        ix.upsert(vectors=to_upsert, namespace=ns)
        print(f"  Upserted {len(to_upsert)} vectors to candidate_id={to_id}")
    except Exception as e:
        print("  Upsert failed:", e)
        sys.exit(1)
    time.sleep(0.2)

print(f"\nFinished upserting {len(all_new_ids)} vectors to {to_id}.")

if args.delete_old:
    confirm = input(f"Delete original {len(found_ids)} vectors with prefix '{from_pref}'? Type DELETE to confirm: ")
    if confirm.strip() == "DELETE":
        for b in chunks(found_ids, args.batch_size):
            try:
                ix.delete(ids=b, namespace=ns)
                print(f"Deleted batch of {len(b)}")
            except Exception as e:
                print("Delete failed:", e)
        print("Deletion done.")
    else:
        print("Deletion aborted by user. Old vectors retained.")
else:
    print("Old vectors retained (no --delete-old). If you want to remove them later rerun with --delete-old.")
