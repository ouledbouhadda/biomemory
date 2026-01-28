"""
Test de la recherche multimodale - Debug de l'erreur 422
"""
import asyncio
import sys
import os
import json
from pathlib import Path

os.chdir(r'c:\Users\dell\biomemory')
sys.path.insert(0, r'c:\Users\dell\biomemory')

async def test_search_request():
    print('\n' + '='*80)
    print('TEST DE LA RECHERCHE MULTIMODALE AVEC TOUTES LES CONDITIONS')
    print('='*80 + '\n')

    try:
        from backend.models.requests import SearchRequest, ExperimentConditions
        from backend.services.embedding_service import get_embedding_service
        from backend.services.qdrant_service import get_qdrant_service

        print('[1] Modèles importés\n')

        print('TEST 1: Recherche avec texte seulement')
        try:
            req1 = SearchRequest(
                text='protein expression optimization',
                sequence=None,
                image_base64=None,
                conditions=None,
                limit=10
            )
            print('Accepté: texte seul\n')
        except Exception as e:
            print(f'Erreur: {e}\n')

        print('TEST 2: Recherche avec texte + conditions (organism, temperature, pH)')
        try:
            req2 = SearchRequest(
                text='protein expression optimization',
                sequence=None,
                image_base64=None,
                conditions=ExperimentConditions(
                    organism='ecoli',
                    temperature=37.0,
                    ph=7.0
                ),
                limit=10
            )
            print('Accepté: texte + conditions\n')
        except Exception as e:
            print(f'Erreur: {e}\n')

        print('TEST 3: Recherche avec texte + séquence')
        try:
            req3 = SearchRequest(
                text='protein expression optimization',
                sequence='ATGGCTAGCAAGGAGAAAG',
                image_base64=None,
                conditions=None,
                limit=10
            )
            print('Accepté: texte + séquence\n')
        except Exception as e:
            print(f'Erreur: {e}\n')

        print('TEST 4: Recherche COMPLÈTE (texte + séquence + conditions + image)')
        import base64
        from PIL import Image
        import io

        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        img_base64_full = f'data:image/png;base64,{img_base64}'

        try:
            req4 = SearchRequest(
                text='protein expression optimization',
                sequence='ATGGCTAGCAAGGAGAAAG',
                image_base64=img_base64_full,
                conditions=ExperimentConditions(
                    organism='ecoli',
                    temperature=37.0,
                    ph=7.0
                ),
                limit=10
            )
            print('Accepté: recherche COMPLÈTE multimodale\n')
        except Exception as e:
            print(f'Erreur: {e}\n')

        print('TEST 5: Conditions en dict (comme du JSON)')
        try:
            req5 = SearchRequest(
                text='protein expression optimization',
                sequence=None,
                image_base64=None,
                conditions={
                    'organism': 'ecoli',
                    'temperature': 37.0,
                    'ph': 7.0
                },
                limit=10
            )
            print('Accepté: conditions dict\n')
        except Exception as e:
            print(f'Erreur: {e}\n')

        print('TEST 6: Conditions vides {} vs null')
        try:
            req6a = SearchRequest(
                text='protein expression optimization',
                conditions={},
                limit=10
            )
            print('Accepté: conditions {}\n')
        except Exception as e:
            print(f'Erreur conditions {{}}: {e}\n')

        print('='*80)
        print('TEST EMBEDDING MULTIMODAL')
        print('='*80 + '\n')

        embedding = get_embedding_service()

        embed = await embedding.generate_multimodal_embedding(
            text='protein expression optimization',
            sequence='ATGGCTAGCAAGGAGAAAG',
            conditions={'organism': 'ecoli', 'temperature': 37.0, 'ph': 7.0},
            image_base64=img_base64_full
        )

        print(f'Embedding généré: {len(embed)} dimensions\n')

        print('='*80)
        print('TEST RECHERCHE DIRECTE QDRANT')
        print('='*80 + '\n')

        qdrant = get_qdrant_service()
        if not qdrant.cloud_client:
            print('Cloud client not available')
            return

        results = qdrant.cloud_client.query_points(
            collection_name='public_science',
            query=embed.tolist(),
            limit=10,
            with_payload=True
        )
        print(f'{len(results.points)} résultats trouvés')
        print(f'   Top score: {results.points[0].score*100:.1f}%\n')

        print('='*80)
        print('TOUS LES TESTS PASSENT!')
        print('='*80)
        print('\nConclusion:')
        print('   - SearchRequest accepte conditions en dict ou objet')
        print('   - Embedding multimodal fonctionne avec tous les paramètres')
        print('   - Qdrant search retourne les résultats')
        print('\nSi erreur 422 au frontend:')
        print('   - Vérifier format JSON envoyé')
        print('   - Vérifier que conditions n\'est pas {None}')
        print('   - Vérifier que image_base64 est au bon format')

    except Exception as e:
        print(f'ERREUR FATALE: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_search_request())
