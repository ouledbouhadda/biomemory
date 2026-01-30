"""
BioMemory Scraper DAG - Runs every 7 days.
Automates protocols.io scraping (API + BeautifulSoup) and post-processing.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "biomemory",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

SCRAPE_QUERIES = [
    "protein expression",
    "protein purification",
    "PCR amplification",
    "western blot",
    "CRISPR genome editing",
    "DNA extraction",
    "RNA extraction",
    "cell culture",
]

SCRAPE_LIMIT_API = 50
SCRAPE_LIMIT_BS4 = 20


def _get_settings():
    import sys, os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from backend.config.settings import get_settings
    return get_settings()


def _get_qdrant_client_and_collection():
    from qdrant_client import QdrantClient
    settings = _get_settings()
    if (
        settings.QDRANT_CLOUD_URL
        and settings.QDRANT_CLOUD_API_KEY
        and settings.QDRANT_CLOUD_URL != "https://your-cluster.qdrant.io"
    ):
        client = QdrantClient(
            url=settings.QDRANT_CLOUD_URL,
            api_key=settings.QDRANT_CLOUD_API_KEY,
        )
    else:
        client = QdrantClient(
            host=settings.QDRANT_PRIVATE_HOST,
            port=settings.QDRANT_PRIVATE_PORT,
        )
    return client, "public_science"


def scrape_api(**context):
    """Scrape protocols.io via API for all queries."""
    import asyncio
    settings = _get_settings()
    from scrapers.protocols_io_scraper import ProtocolsIOScraper
    from scrapers.storage_service import ScrapedExperimentStorage

    scraper = ProtocolsIOScraper(api_token=settings.PROTOCOLS_IO_API_TOKEN)
    client, collection = _get_qdrant_client_and_collection()
    storage = ScrapedExperimentStorage(client, collection_name=collection)

    total = 0
    for query in SCRAPE_QUERIES:
        print(f"[API] Scraping: {query}")
        experiments = scraper.scrape(query=query, limit=SCRAPE_LIMIT_API)
        if experiments:
            asyncio.run(storage.store_batch(experiments))
            total += len(experiments)
        print(f"[API] {query}: {len(experiments)} experiments stored")

    print(f"[API] Total: {total} experiments")
    return total


def scrape_bs4(**context):
    """Scrape protocols.io via BeautifulSoup for all queries."""
    import asyncio
    settings = _get_settings()
    from scrapers.protocols_io_scraper import ProtocolsIOScraper
    from scrapers.storage_service import ScrapedExperimentStorage

    scraper = ProtocolsIOScraper(api_token=settings.PROTOCOLS_IO_API_TOKEN)
    client, collection = _get_qdrant_client_and_collection()
    storage = ScrapedExperimentStorage(client, collection_name=collection)

    total = 0
    for query in SCRAPE_QUERIES:
        print(f"[BS4] Scraping: {query}")
        experiments = scraper.scrape_bs4(query=query, limit=SCRAPE_LIMIT_BS4)
        if experiments:
            asyncio.run(storage.store_batch(experiments))
            total += len(experiments)
        print(f"[BS4] {query}: {len(experiments)} experiments stored")

    print(f"[BS4] Total: {total} experiments")
    return total


def post_process_success_field(**context):
    """Add success field (70% True / 30% False) to new points without it."""
    import random
    from qdrant_client.models import FieldCondition, MatchValue, Filter

    client, collection = _get_qdrant_client_and_collection()
    offset = None
    updated = 0

    while True:
        results = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, offset = results
        if not points:
            break

        ids_to_update = []
        for point in points:
            if "success" not in point.payload:
                ids_to_update.append(point.id)

        for pid in ids_to_update:
            client.set_payload(
                collection_name=collection,
                payload={"success": random.random() < 0.7},
                points=[pid],
            )
            updated += 1

        if offset is None:
            break

    print(f"[PostProcess] Updated success field on {updated} points")
    return updated


def mark_crispr_failed(**context):
    """Mark CRISPR-related experiments as failed."""
    client, collection = _get_qdrant_client_and_collection()
    offset = None
    marked = 0
    keywords = ["crispr", "cas9", "cas12", "cas13", "guide rna", "grna", "sgrna"]

    while True:
        results = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, offset = results
        if not points:
            break

        for point in points:
            text = (
                point.payload.get("text", "")
                + " "
                + point.payload.get("title", "")
            ).lower()
            if any(kw in text for kw in keywords):
                client.set_payload(
                    collection_name=collection,
                    payload={"success": False},
                    points=[point.id],
                )
                marked += 1

        if offset is None:
            break

    print(f"[PostProcess] Marked {marked} CRISPR experiments as failed")
    return marked


with DAG(
    dag_id="biomemory_scraper",
    default_args=default_args,
    description="Scrape protocols.io (API + BeautifulSoup) every 7 days",
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["biomemory", "scraper", "protocols.io"],
) as dag:

    task_scrape_api = PythonOperator(
        task_id="scrape_protocols_api",
        python_callable=scrape_api,
    )

    task_scrape_bs4 = PythonOperator(
        task_id="scrape_protocols_bs4",
        python_callable=scrape_bs4,
    )

    task_success_field = PythonOperator(
        task_id="post_process_success_field",
        python_callable=post_process_success_field,
    )

    task_crispr = PythonOperator(
        task_id="mark_crispr_failed",
        python_callable=mark_crispr_failed,
    )

    # API and BS4 scraping run in parallel, then post-processing
    [task_scrape_api, task_scrape_bs4] >> task_success_field >> task_crispr
