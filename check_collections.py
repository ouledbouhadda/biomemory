import asyncio
from backend.services.qdrant_service import get_qdrant_service
async def check():
    qdrant = get_qdrant_service()
    try:
        results = await qdrant.scroll('private_experiments', limit=100)
        print(f'Private experiments: {len(results)} points')
    except Exception as e:
        print(f'Private check error: {e}')
    try:
        results = await qdrant.scroll('public_science', limit=100)
        print(f'Cloud public_science: {len(results)} points')
    except Exception as e:
        print(f'Cloud check error: {e}')
if __name__ == "__main__":
    asyncio.run(check())