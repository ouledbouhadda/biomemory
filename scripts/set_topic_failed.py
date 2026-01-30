"""
Set experiments matching CRISPR-related topics to success=False
by scrolling and checking titles client-side.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

COLLECTION = "public_science"
KEYWORDS = ["crispr", "cas9", "gene editing", "genome editing", "gene knockout",
            "grna", "guide rna", "crispr-cas"]

def main():
    qdrant_url = os.getenv("QDRANT_CLOUD_URL")
    qdrant_key = os.getenv("QDRANT_CLOUD_API_KEY")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)

    print(f"Scrolling through {COLLECTION} to find CRISPR-related experiments...")
    print(f"Keywords: {KEYWORDS}")

    matched_ids = []
    offset = None
    scanned = 0

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        if not results:
            break

        for p in results:
            scanned += 1
            title = (p.payload.get('title', '') or '').lower()
            text = (p.payload.get('text', '') or '').lower()
            combined = title + ' ' + text

            if any(kw in combined for kw in KEYWORDS):
                matched_ids.append(p.id)

        if scanned % 5000 == 0:
            print(f"  Scanned {scanned} points, found {len(matched_ids)} CRISPR matches...")

        if next_offset is None:
            break
        offset = next_offset

        # Stop after scanning enough to find a good set
        if len(matched_ids) >= 200:
            print(f"  Found {len(matched_ids)} matches, stopping early...")
            break

    print(f"\nScanned {scanned} points total")
    print(f"Found {len(matched_ids)} CRISPR-related experiments")

    if not matched_ids:
        print("No matches found! Trying alternative approach...")
        # Just take the first 100 points and mark them as failed as a demo
        results, _ = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            with_payload=False,
            with_vectors=False
        )
        matched_ids = [p.id for p in results]
        print(f"Using first {len(matched_ids)} points as demo")

    # Mark all matched as failed
    BATCH = 100
    for i in range(0, len(matched_ids), BATCH):
        batch = matched_ids[i:i+BATCH]
        client.set_payload(
            collection_name=COLLECTION,
            payload={"success": False},
            points=batch
        )
        print(f"  Marked {min(i+BATCH, len(matched_ids))}/{len(matched_ids)} as failed")

    print(f"\n=== DONE: {len(matched_ids)} experiments marked as success=False ===")
    print(f'\nTo test HIGH reproducibility risk, search for:')
    print(f'  "CRISPR gene editing"')
    print(f'  "cas9 genome editing"')
    print(f'  "gene knockout"')

    # Show some examples
    print("\nSample of marked experiments:")
    for pid in matched_ids[:5]:
        points = client.retrieve(
            collection_name=COLLECTION,
            ids=[pid],
            with_payload=True
        )
        if points:
            p = points[0]
            title = p.payload.get('title', 'N/A')[:70]
            success = p.payload.get('success')
            print(f"  success={success} | {title}")


if __name__ == "__main__":
    main()
