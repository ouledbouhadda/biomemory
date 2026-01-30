from qdrant_client import QdrantClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.protocols_io_scraper import ProtocolsIOScraper
from scrapers.storage_service import ScrapedExperimentStorage
from backend.config.settings import get_settings
from backend.services.qdrant_service import QdrantService
import asyncio
settings = get_settings()
async def main():
    print(" Starting protocols.io scraper...")
    scraper = ProtocolsIOScraper(api_token=settings.PROTOCOLS_IO_API_TOKEN)
    # Method 1: API-based scraping
    experiments = scraper.scrape(
        query="protein expression",
        limit=50
    )
    print(f"API scraper: {len(experiments)} experiments")

    # Method 2: BeautifulSoup HTML scraping for additional data
    bs4_experiments = scraper.scrape_bs4(
        query="protein purification",
        limit=20
    )
    print(f"BS4 scraper: {len(bs4_experiments)} experiments")

    experiments.extend(bs4_experiments)

    if not experiments:
        print("No experiments found")
        return
    qdrant_service = QdrantService()
    await qdrant_service.init_collections()
    if settings.QDRANT_CLOUD_URL and settings.QDRANT_CLOUD_API_KEY and settings.QDRANT_CLOUD_URL != "https://your-cluster.qdrant.io":
        print("Using Qdrant Cloud for public data")
        qdrant = QdrantClient(
            url=settings.QDRANT_CLOUD_URL,
            api_key=settings.QDRANT_CLOUD_API_KEY
        )
        collection_name = "public_science"
        print(f"Connected to Qdrant Cloud: {settings.QDRANT_CLOUD_URL}")
    else:
        print("Qdrant Cloud not configured, using local Qdrant instance for public data")
        qdrant = QdrantClient(
            host=settings.QDRANT_PRIVATE_HOST,
            port=settings.QDRANT_PRIVATE_PORT
        )
        collection_name = "public_science"
        try:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config={"size": 768, "distance": "Cosine"}
            )
            print(f"Created collection '{collection_name}'")
        except Exception as e:
            print(f"Collection '{collection_name}' already exists or error: {e}")
    storage = ScrapedExperimentStorage(qdrant, collection_name=collection_name)
    await storage.store_batch(experiments)
    print("Done!")
if __name__ == "__main__":
    asyncio.run(main())