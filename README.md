# BioMemory

**Multimodal Biological Design & Discovery Intelligence System**

BioMemory est une plateforme d'intelligence artificielle qui aide les chercheurs en biologie a rechercher, analyser et optimiser des protocoles experimentaux. Le systeme combine recherche multimodale par vecteurs, generation de conceptions assistee par IA (RAG), scraping automatise de protocoles scientifiques et analyse de reproductibilite.

L'objectif principal : permettre aux chercheurs de ne plus repeter des experiences ratees sans le savoir, en construisant une **memoire collective** de la biologie experimentale.

---

## Project Overview & Objectives

### Probleme
Les protocoles scientifiques sont disperses, non structures, et les chercheurs n'ont aucun moyen simple de savoir si une experience similaire a deja ete tentee (et si elle a reussi ou echoue).

### Solution
BioMemory centralise les protocoles experimentaux dans une base de donnees vectorielle (Qdrant), les encode en **embeddings multimodaux 488D** (texte + sequence biologique + conditions), et permet :

1. **Recherche semantique multimodale** : trouver des protocoles similaires par texte, sequence ADN/ARN/proteine, conditions experimentales ou image
2. **Generation de variantes (RAG)** : proposer des optimisations de protocoles basees sur des experiences passees similaires, via Google Gemini
3. **Analyse de reproductibilite** : evaluer la probabilite de succes d'une experience avant de l'executer
4. **Scraping automatise** : enrichir continuellement la base via protocols.io (API + BeautifulSoup), orchestre par Apache Airflow
5. **Planification agentique** : generer des plans experimentaux complets avec etapes detaillees

---

## Technologies Used

### Backend
| Technologie | Version | Role |
|---|---|---|
| Python | 3.10+ | Langage principal |
| FastAPI | 0.104.1 | Framework API REST |
| Pydantic | 2.5.0 | Validation des donnees |
| Uvicorn | 0.24.0 | Serveur ASGI |

### Intelligence Artificielle
| Technologie | Version | Role |
|---|---|---|
| SentenceTransformers (all-MiniLM-L6-v2) | 2.2.2 | Embeddings texte (384D) |
| CLIP (openai/clip-vit-base-patch32) | via transformers | Embeddings image (512D) |
| Google Gemini | gemini-1.5-pro | Generation RAG (variantes, planification) |
| Groq (Llama 3 70B) | API | Alternative LLM pour inference rapide |
| PyTorch | 2.1.0 | Runtime pour les modeles |

### Base de Donnees Vectorielle
| Technologie | Version | Role |
|---|---|---|
| Qdrant | >=1.7.1 (client) | Stockage et recherche vectorielle |
| Qdrant Cloud | - | Instance publique (public_science) |
| Qdrant Local (Docker) | - | Instance privee (private_experiments) |

### Frontend
| Technologie | Version | Role |
|---|---|---|
| React | 18.3.0 | Framework UI |
| Vite | 5.4.0 | Build tool |
| Axios | 1.7.0 | Client HTTP |
| Lucide React | 0.400.0 | Icones |

### Scraping & Data Pipeline
| Technologie | Version | Role |
|---|---|---|
| BeautifulSoup4 | 4.12.2 | Scraping HTML |
| Requests | 2.31.0 | Requetes HTTP |
| httpx | 0.25.2 | Client HTTP async |
| Apache Airflow | 2.7.3 | Orchestration des taches periodiques |

### Securite
| Technologie | Version | Role |
|---|---|---|
| python-jose | 3.3.0 | JWT tokens |
| passlib (bcrypt) | 1.7.4 | Hachage mots de passe |
| cryptography | 41.0.7 | Chiffrement des donnees |
| pyotp | 2.9.0 | Authentification MFA (TOTP) |

### Bio Libraries
| Technologie | Version | Role |
|---|---|---|
| BioPython | 1.81 | Manipulation de sequences biologiques |
| NumPy | 1.24.3 | Calcul numerique |
| scikit-learn | 1.3.2 | Metriques de similarite |

### Monitoring & Testing
| Technologie | Version | Role |
|---|---|---|
| Prometheus Client | 0.19.0 | Metriques d'observabilite |
| pytest | 7.4.3 | Tests unitaires |
| pytest-asyncio | 0.21.1 | Tests async |

---

## Project Architecture

```
biomemory/
|
|-- backend/                          # API REST (FastAPI)
|   |-- main.py                       # Point d'entree, lifespan, middlewares
|   |-- config/
|   |   |-- settings.py               # Configuration centralisee (Pydantic Settings)
|   |
|   |-- agents/                       # Pipeline multi-agents
|   |   |-- orchestrator.py           # Coordonne tous les agents
|   |   |-- ingestion_agent.py        # Nettoyage et normalisation des donnees
|   |   |-- chunking_agent.py         # Segmentation de texte en chunks
|   |   |-- embedding_agent.py        # Generation d'embeddings
|   |   |-- multimodal_search_agent.py # Detection des modalites (texte/seq/image)
|   |   |-- similarity_agent.py       # Recherche de similarite dans Qdrant
|   |   |-- reranking_agent.py        # Reranking RRF + boosting temporel
|   |   |-- design_agent.py           # Generation de variantes de conception
|   |   |-- evidence_agent.py         # Construction de preuves corroboratives
|   |   |-- failure_agent.py          # Analyse des risques de reproductibilite
|   |   |-- bio_rag_agent.py          # Reponses generatives RAG
|   |   |-- agentic_rag_agent.py      # Planification experimentale agentique
|   |
|   |-- services/                     # Services metier
|   |   |-- qdrant_service.py         # Interface Qdrant (Cloud + Local)
|   |   |-- embedding_service.py      # Embeddings multimodaux (384+100+4=488D)
|   |   |-- gemini_service.py         # Integration Google Gemini
|   |   |-- groq_service.py           # Alternative Groq/Llama
|   |   |-- bio_rag_service.py        # Service BioRAG
|   |   |-- chunking_service.py       # Strategies de chunking
|   |
|   |-- api/routes/                   # Endpoints REST
|   |   |-- auth.py                   # Authentification JWT + MFA
|   |   |-- experiments.py            # Upload et gestion d'experiences
|   |   |-- search.py                 # Recherche multimodale + agentique
|   |   |-- design.py                 # Generation de variantes
|   |   |-- health.py                 # Health check
|   |
|   |-- models/                       # Schemas Pydantic
|   |   |-- scraped_experiment.py     # Modele d'experience scrapee
|   |   |-- requests.py              # Modeles de requetes
|   |   |-- responses.py             # Modeles de reponses
|   |
|   |-- security/                     # Couche securite
|   |   |-- auth.py                   # Logique d'authentification
|   |   |-- audit_logger.py           # Journalisation d'audit
|   |   |-- encryption.py            # Chiffrement AES
|   |   |-- mfa.py                    # Multi-Factor Authentication
|   |   |-- rate_limiting.py          # Rate limiting par IP/utilisateur
|   |
|   |-- database/                     # Base de donnees relationnelle
|       |-- user_repository.py        # Gestion des utilisateurs (SQLite)
|
|-- frontend/                         # Interface React
|   |-- src/components/
|   |   |-- Dashboard.jsx             # Vue d'ensemble + statistiques
|   |   |-- SearchPanel.jsx           # Recherche multimodale
|   |   |-- ExperimentUpload.jsx      # Formulaire d'upload
|   |   |-- SimilarExperiments.jsx    # Resultats de recherche
|   |   |-- DesignSuggestions.jsx     # Variantes de conception
|   |   |-- AgenticPlanning.jsx       # Planification experimentale IA
|   |   |-- ReproducibilityRisk.jsx   # Analyse de risque
|   |   |-- DnaHelix.jsx             # Animation ADN
|
|-- scrapers/                         # Extraction de donnees
|   |-- protocols_io_scraper.py       # Scraper protocols.io (API + BeautifulSoup)
|   |-- storage_service.py           # Stockage batch dans Qdrant
|
|-- airflow/dags/                     # Workflows automatises
|   |-- biomemory_scraper_dag.py      # DAG hebdomadaire de scraping
|
|-- scripts/                          # Scripts utilitaires
|   |-- run_protocols_scraper.py      # Execution manuelle du scraper
|
|-- requirements.txt                  # Dependances Python
```

### Diagramme d'architecture

```
                         +------------------+
                         |    Frontend      |
                         |   React + Vite   |
                         +--------+---------+
                                  |
                            HTTP/REST API
                                  |
                    +-------------v--------------+
                    |      FastAPI Backend        |
                    |  (Middlewares: CORS, Rate   |
                    |   Limit, Audit, JWT Auth)   |
                    +-------------+--------------+
                                  |
              +-------------------+-------------------+
              |                   |                   |
     +--------v------+  +--------v------+  +---------v-------+
     | Orchestrator   |  | Design Route  |  | Search Route    |
     | Agent          |  | (RAG direct)  |  | (Multimodal)    |
     +--------+------+  +--------+------+  +---------+-------+
              |                   |                   |
    +---------v---------+        |          +---------v--------+
    | Pipeline Agents   |        |          | EmbeddingService |
    | Ingestion         |        |          | (488D multimodal)|
    | Chunking          |        |          +---------+--------+
    | Embedding         |        |                    |
    | Similarity        |        |        +-----------v-----------+
    | Reranking         |        |        |                       |
    | Evidence          |        |        v                       v
    | Failure           |        |   +---------+          +------------+
    +--------+----------+        |   | Qdrant  |          | Qdrant     |
             |                   |   | Cloud   |          | Local      |
             v                   |   | (public)|          | (private)  |
    +--------+----------+       |   +---------+          +------------+
    | Gemini / Groq LLM |<------+
    +-------------------+
              ^
              |
    +---------+---------+
    | Airflow DAG       |
    | (scraping weekly) |
    | protocols.io      |
    | API + BS4         |
    +-------------------+
```

---

## Integration Detaillee de Qdrant

Qdrant est le composant central de BioMemory pour le stockage et la recherche vectorielle. Voici comment il est integre en detail :

### 1. Architecture Dual-Instance

BioMemory utilise **deux instances Qdrant** :

| Instance | Collection | Usage | Connexion |
|---|---|---|---|
| **Qdrant Cloud** | `public_science` | Protocoles publics scrapes (protocols.io) | HTTPS + API Key + gRPC |
| **Qdrant Local** (Docker) | `private_experiments` | Experiences privees des utilisateurs | localhost:6333 |

Le routage est automatique via `_get_client()` dans [qdrant_service.py](backend/services/qdrant_service.py) : selon le nom de la collection, le bon client est selectionne.

### 2. Configuration HNSW (Hierarchical Navigable Small World)

L'algorithme HNSW est **explicitement configure** pour optimiser la recherche ANN (Approximate Nearest Neighbor) :

```python
hnsw_config = {
    "m": 32,                    # Nombre de connexions par noeud (precision vs memoire)
    "ef_construct": 200,        # Qualite de construction de l'index (plus = meilleur index)
    "full_scan_threshold": 10000, # Seuil en dessous duquel un scan lineaire est utilise
    "max_indexing_threads": 4,  # Parallelisme de l'indexation
    "on_disk": False            # Index en RAM pour des performances maximales
}
```

**Comment HNSW fonctionne dans BioMemory :**
- Chaque experience est encodee en un vecteur 488D (384 texte + 100 sequence + 4 conditions)
- HNSW construit un graphe multi-couches navigable pour trouver les voisins les plus proches
- La recherche se fait en **O(log n)** au lieu de O(n) pour un scan lineaire
- Avec `m=32`, chaque noeud a 32 connexions, offrant un bon equilibre precision/vitesse
- `ef_construct=200` garantit un index de haute qualite lors de l'insertion

### 3. Quantization INT8

Pour reduire l'empreinte memoire sans perte significative de precision :

```python
quantization_config = {
    "scalar": {
        "type": "int8",       # Reduction de float32 a int8 (4x moins de memoire)
        "quantile": 0.99,     # Calibration sur le 99e percentile
        "always_ram": True    # Vecteurs quantifies toujours en RAM
    }
}
```

### 4. Indexation des Payloads

Des index sont crees sur les champs de metadonnees pour le filtrage rapide :

- **TEXT** : `title`, `abstract`, `text` (tokenization par mots, lowercase)
- **KEYWORD** : `domain`, `experiment_type`, `organism`, `conditions.organism`
- **FLOAT** : `temperature`, `ph`, `confidence_score`, `score`
- **DATETIME** : `publication_date`, `scraped_at`

### 5. Modes de Recherche

| Mode | Methode | Description |
|---|---|---|
| **Vectorielle** | `search()` | Recherche cosinus standard dans l'espace 488D |
| **Hybride** | `hybrid_search()` | Fusion RRF de recherche vectorielle + textuelle (Prefetch) |
| **Hybride Sparse** | `hybrid_search_with_sparse()` | Dense + sparse vectors (BM25-like) avec fusion RRF |
| **Temporelle** | `search_temporal()` | Recherche vectorielle filtree par plage de dates |
| **Avec Grouping** | `search_with_grouping()` | Resultats groupes par champ (ex: organisme) |
| **Avec Boosting** | `search_with_boosting()` | Re-scoring base sur des facteurs de boost |
| **Batch** | `batch_search()` | Recherche parallele de multiples queries |
| **Recommend** | `recommend()` | Recommandation par exemples positifs/negatifs |
| **Discover** | `discover()` | Decouverte contextuelle avec paires d'exemples |

### 6. Resilience

- **Circuit Breaker** : Si Qdrant Cloud echoue 5 fois consecutives, le systeme bascule automatiquement sur l'instance locale pendant 60 secondes
- **Cache TTL-LRU** : 4 caches separees (search, stats, facets, scroll) avec TTL de 5-10 minutes et taille max de 64-256 entrees
- **Snapshots** : Creation et restauration de backups de collections
- **Aliases** : Support blue-green deployment via aliases de collections

### 7. Flux de Donnees Complet

```
1. INGESTION
   Utilisateur/Scraper -> IngestionAgent -> Nettoyage/Validation
                                              |
2. CHUNKING                                   v
   ChunkingAgent -> Segmentation intelligente (auto/biological/paragraph)
                                              |
3. EMBEDDING                                  v
   EmbeddingService -> [384D texte] + [100D sequence k-mer] + [4D conditions]
                     = Vecteur 488D normalise
                                              |
4. STOCKAGE                                   v
   QdrantService.upsert() -> Qdrant (avec HNSW indexation + payload indexes)
                                              |
5. RECHERCHE                                  v
   Query -> EmbeddingService -> QdrantService.search() -> HNSW O(log n)
                                              |
6. POST-TRAITEMENT                            v
   SimilarityAgent -> RerankingAgent (RRF + temporal boost) -> EvidenceAgent
                                              |
7. GENERATION (RAG)                           v
   Contexte Qdrant + Prompt -> Gemini -> Reponse generee
```

---

## Setup and Installation

### Prerequis

- **Python 3.10+**
- **Node.js 18+**
- **Docker & Docker Compose** (pour Qdrant local)
- **Cle API Google Gemini** (pour la generation RAG)
- **Compte Qdrant Cloud** (optionnel, pour les donnees publiques)

### Etape 1 : Cloner le projet

```bash
git clone https://github.com/your-username/biomemory.git
cd biomemory
```

### Etape 2 : Creer un environnement virtuel Python

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Etape 3 : Installer les dependances Python

```bash
pip install -r requirements.txt
```

> **Note** : L'installation de `torch` et `sentence-transformers` peut prendre du temps (~2 Go).

### Etape 4 : Demarrer Qdrant (Docker)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verifier que Qdrant est actif :
```bash
curl http://localhost:6333/healthz
# Attendu: {"title":"qdrant - vectorass engine","version":"..."}
```

### Etape 5 : Configurer les variables d'environnement

Creer un fichier `.env` a la racine du projet :

```env
# --- Securite ---
SECRET_KEY=votre_cle_secrete_longue_et_aleatoire
ENCRYPTION_KEY=votre_cle_de_chiffrement_32_chars

# --- Google Gemini ---
GEMINI_API_KEY=votre_cle_api_gemini

# --- Qdrant Local (obligatoire) ---
QDRANT_PRIVATE_HOST=localhost
QDRANT_PRIVATE_PORT=6333

# --- Qdrant Cloud (optionnel) ---
QDRANT_CLOUD_URL=https://votre-cluster.cloud.qdrant.io
QDRANT_CLOUD_API_KEY=votre_cle_api_qdrant_cloud

# --- Groq (optionnel) ---
GROQ_API_KEY=votre_cle_groq
USE_GROQ=false

# --- Protocols.io (optionnel, pour le scraping) ---
PROTOCOLS_IO_API_TOKEN=votre_token_protocols_io

# --- Configuration ---
DEBUG=true
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:5173"]
DATABASE_URL=sqlite:///./biomemory.db
```

### Etape 6 : Lancer le backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Verifier que l'API est active :
- Documentation Swagger : http://localhost:8000/api/docs
- Documentation ReDoc : http://localhost:8000/api/redoc
- Health check : http://localhost:8000/api/v1/health

Au demarrage, le backend :
1. Initialise les connexions Qdrant (Cloud + Local)
2. Cree les collections avec la configuration HNSW et les index de payload
3. Charge le modele SentenceTransformer `all-MiniLM-L6-v2`

### Etape 7 : Installer et lancer le frontend

```bash
cd frontend
npm install
npm run dev
```

L'interface sera disponible sur http://localhost:5173

### Etape 8 (optionnel) : Executer le scraper

Pour peupler la base avec des protocoles publics :

```bash
# Execution manuelle
python scripts/run_protocols_scraper.py

# Ou via Airflow (execution automatique toutes les semaines)
# Configurer Airflow et placer le DAG dans airflow/dags/
```

### Verification de l'installation

1. **Backend** : Acceder a http://localhost:8000/api/docs et tester l'endpoint `GET /api/v1/health`
2. **Qdrant** : Acceder a http://localhost:6333/dashboard pour voir les collections
3. **Frontend** : Acceder a http://localhost:5173 et verifier que le Dashboard se charge
4. **Creer un compte** : S'inscrire via l'interface ou via `POST /api/v1/auth/register`
5. **Tester une recherche** : Saisir "protein expression E. coli" dans le SearchPanel

---

## API Endpoints

### Authentification
| Methode | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Inscription utilisateur |
| POST | `/api/v1/auth/login` | Connexion (retourne JWT) |

### Experiences
| Methode | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/experiments/upload` | Telecharger une experience |
| GET | `/api/v1/experiments/stats` | Statistiques du systeme |

### Recherche
| Methode | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/search/multimodal` | Recherche multimodale |
| POST | `/api/v1/search/agentic-planning` | Planification agentique |
| GET | `/api/v1/search/facets/{field}` | Facettes pour filtres |

### Conception
| Methode | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/design/variants` | Generer des variantes optimisees |

### Sante
| Methode | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Etat du systeme et des connexions |

---

## Securite

- **JWT** avec expiration configurable (30 min par defaut)
- **bcrypt** pour le hachage des mots de passe
- **MFA/TOTP** avec generation de QR code
- **Rate limiting** par IP et par utilisateur (100 req/min global, 10 req/min design)
- **Audit logging** complet de toutes les actions
- **Chiffrement AES** des donnees sensibles
- **CORS** restrictif en production
- **TrustedHost** middleware en production
- **Separation des donnees** : collections publiques (Cloud) vs privees (Local)
