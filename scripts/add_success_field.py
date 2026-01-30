"""
Script to add 'success' field to all points in the public_science collection.
Distribution: 70% success (True), 30% failed (False)
Uses batch updates for performance.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import SetPayloadOperation, SetPayload, PointIdsList

COLLECTION = "public_science"
SUCCESS_RATE = 0.70
SCROLL_BATCH = 100

def main():
    qdrant_url = os.getenv("QDRANT_CLOUD_URL")
    qdrant_key = os.getenv("QDRANT_CLOUD_API_KEY")

    if not qdrant_url or not qdrant_key:
        print("ERROR: QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY must be set in .env")
        sys.exit(1)

    print(f"Connecting to Qdrant Cloud: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)

    collections = client.get_collections()
    col_names = [c.name for c in collections.collections]
    print(f"Available collections: {col_names}")

    if COLLECTION not in col_names:
        print(f"ERROR: Collection '{COLLECTION}' not found!")
        sys.exit(1)

    info = client.get_collection(COLLECTION)
    total_points = info.points_count
    print(f"Collection '{COLLECTION}': {total_points} points")
    print(f"Distribution: {SUCCESS_RATE*100:.0f}% success / {(1-SUCCESS_RATE)*100:.0f}% failed")
    print()

    # Collect all point IDs first
    all_ids = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )
        if not results:
            break
        for point in results:
            all_ids.append(point.id)
        print(f"  Collected {len(all_ids)} point IDs...")
        if next_offset is None:
            break
        offset = next_offset

    print(f"\nTotal points collected: {len(all_ids)}")

    # Assign random success values
    success_ids = []
    failed_ids = []
    for pid in all_ids:
        if random.random() < SUCCESS_RATE:
            success_ids.append(pid)
        else:
            failed_ids.append(pid)

    print(f"Success: {len(success_ids)} ({len(success_ids)/len(all_ids)*100:.1f}%)")
    print(f"Failed:  {len(failed_ids)} ({len(failed_ids)/len(all_ids)*100:.1f}%)")

    # Batch update - set success=True for success_ids
    BATCH = 100
    print(f"\nUpdating success=True for {len(success_ids)} points...")
    for i in range(0, len(success_ids), BATCH):
        batch = success_ids[i:i+BATCH]
        client.set_payload(
            collection_name=COLLECTION,
            payload={"success": True},
            points=batch
        )
        print(f"  Updated {min(i+BATCH, len(success_ids))}/{len(success_ids)}")

    # Batch update - set success=False for failed_ids
    print(f"\nUpdating success=False for {len(failed_ids)} points...")
    for i in range(0, len(failed_ids), BATCH):
        batch = failed_ids[i:i+BATCH]
        client.set_payload(
            collection_name=COLLECTION,
            payload={"success": False},
            points=batch
        )
        print(f"  Updated {min(i+BATCH, len(failed_ids))}/{len(failed_ids)}")

    # Verify
    print("\n--- Verification ---")
    sample, _ = client.scroll(
        collection_name=COLLECTION,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    s_count = 0
    f_count = 0
    for p in sample:
        val = p.payload.get('success', 'MISSING')
        title = p.payload.get('title', 'N/A')[:50]
        status = "OK" if val in (True, False) else "MISSING"
        print(f"  ID={p.id} | success={val} | {title}")
        if val is True:
            s_count += 1
        elif val is False:
            f_count += 1

    print(f"\nSample: {s_count} success, {f_count} failed out of {len(sample)}")
    print("DONE!")


if __name__ == "__main__":
    main()
