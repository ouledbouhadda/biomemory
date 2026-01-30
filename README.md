# BioMemory: Multimodal Biological Design & Discovery Intelligence System

BioMemory est une plateforme d'intelligence artificielle avancée conçue pour révolutionner la recherche biologique en combinant recherche multimodale, génération de conceptions assistée par IA et analyse de reproductibilité. Le système utilise des techniques de pointe en apprentissage automatique pour aider les chercheurs à découvrir, analyser et optimiser des protocoles expérimentaux biologiques.

## Vue d'ensemble de l'architecture

BioMemory est organisé autour de plusieurs composants clés qui travaillent ensemble pour fournir une expérience de recherche complète :

### Backend (FastAPI)
Le backend fournit une API REST robuste avec les fonctionnalités suivantes :

#### Agents d'IA
- **OrchestratorAgent** : Coordonne l'exécution des pipelines de recherche et de conception
- **IngestionAgent** : Nettoie et normalise les données expérimentales entrantes
- **ChunkingAgent** : Divise les textes longs en segments significatifs pour l'indexation
- **EmbeddingAgent** : Génère des représentations vectorielles des expériences
- **MultimodalSearchAgent** : Détecte et traite les requêtes multimodales (texte, séquences, images)
- **SimilarityAgent** : Effectue des recherches de similarité avec filtrage avancé
- **RerankingAgent** : Réorganise les résultats de recherche avec fusion RRF et boosting temporel
- **DesignAgent** : Génère des variantes de conception optimisées
- **EvidenceAgent** : Construit des preuves corroboratives pour les résultats
- **FailureAgent** : Analyse les risques de reproductibilité et les patterns d'échec
- **BioRAGAgent** : Fournit des réponses génératives augmentées par récupération pour les questions biologiques

#### Services
- **QdrantService** : Interface avec la base de données vectorielle Qdrant
- **EmbeddingService** : Génère des embeddings multimodaux avec cache LRU
- **GeminiService** : Intégration avec Google Gemini pour la génération de contenu
- **GroqService** : Alternative basée sur Groq pour l'inférence rapide
- **BioRAGService** : Service spécialisé utilisant BioBERT et BioGPT
- **ChunkingService** : Segmentation intelligente de texte avec stratégies biologiques

### Frontend (React)
Interface utilisateur moderne avec :
- **Dashboard** : Vue d'ensemble avec statistiques et santé du système
- **SearchPanel** : Recherche multimodale avec filtres avancés
- **ExperimentUpload** : Téléchargement d'expériences avec validation
- **SimilarExperiments** : Affichage enrichi des résultats de recherche
- **DesignSuggestions** : Présentation des variantes de conception générées
- **AgenticPlanning** : Planification expérimentale assistée par IA
- **ReproducibilityRisk** : Analyse des risques de reproductibilité

### Base de données vectorielle : Qdrant

Qdrant joue un rôle central dans BioMemory en tant que base de données vectorielle spécialisée :

#### Collections
- **public_science** : Stocke les expériences scrapées depuis protocols.io et autres sources publiques
- **biomemory_users** : Gestion des utilisateurs et authentification
- **private_experiments** : Expériences utilisateur privées avec accès contrôlé

#### Fonctionnalités clés
- **Recherche de similarité** : Recherche cosinus dans l'espace vectoriel 768D
- **Filtrage avancé** : Conditions sur domaine, type d'expérience, date, confiance
- **Mise à jour en temps réel** : Indexation continue des nouvelles expériences
- **Recherche hybride** : Combinaison de recherche vectorielle et textuelle
- **Recherche multimodale** : Gestion des embeddings texte + séquence + conditions

#### Flux de données dans Qdrant
1. **Ingestion** : Les expériences sont nettoyées, chunkées et embeddées
2. **Indexation** : Vecteurs stockés avec métadonnées enrichies
3. **Recherche** : Requêtes transformées en vecteurs et comparées
4. **Récupération** : Résultats rerankés et enrichis avec preuves
5. **Génération** : Contextes utilisés pour RAG et conception

### Scrapers et ingestion de données
- **Scraper** : Extraction automatisée(beautifulsoup4)
- **StorageService** : Gestion du stockage batch dans Qdrant
- **Workflows Airflow** : Orchestration des tâches de scraping périodiques(une fois/semaine)

## Fonctionnalités principales

### 1. Recherche multimodale
**Comment ça fonctionne** : L'utilisateur fournit du texte, une séquence biologique, des conditions expérimentales ou une image. Le système détecte automatiquement les modalités présentes et génère un embedding adaptatif.

**Ce que ça donne** : Liste de protocoles similaires avec scores de similarité, preuves corroboratives et métriques de confiance.

**À quoi ça sert** : Découvrir des protocoles existants similaires à une idée expérimentale, éviter la réinvention et identifier des optimisations potentielles.

**Flux de données** :
- Requête utilisateur → MultimodalSearchAgent → EmbeddingService → Qdrant search → SimilarityAgent → RerankingAgent → EvidenceAgent → Résultats enrichis

### 2. Téléchargement d'expériences
**Comment ça fonctionne** : Interface de formulaire pour saisir description textuelle, séquence, conditions (organisme, température, pH) et résultat (succès/échec).

**Ce que ça donne** : ID unique d'expérience et confirmation d'indexation.

**À quoi ça sert** : Construire une base de connaissances personnelle et contribuer à la communauté scientifique.

**Flux de données** :
- Formulaire → IngestionAgent → ChunkingAgent → EmbeddingAgent → Qdrant upsert dans private_experiments

### 3. Génération de variantes de conception
**Comment ça fonctionne** : À partir d'expériences similaires réussies et échouées, Gemini analyse les patterns et génère des modifications optimisées.

**Ce que ça donne** : Liste de variantes avec justifications, niveaux de confiance et séquences modifiées.

**À quoi ça sert** : Accélérer l'optimisation de protocoles en suggérant des modifications basées sur des données réelles.

**Flux de données** :
- Requête de conception → Orchestrator → Similarity search → DesignAgent (avec Gemini) → Variants justifiées

### 4. Planification expérimentale agentique
**Comment ça fonctionne** : Pipeline RAG spécialisé qui récupère des expériences pertinentes et génère des plans complets avec étapes détaillées.

**Ce que ça donne** : Protocole complet avec variantes multiples, risques identifiés et recommandations.

**À quoi ça sert** : Aider les chercheurs novices ou dans de nouveaux domaines à planifier des expériences robustes.

**Flux de données** :
- Description de but → BioRAGAgent → Qdrant retrieval → BioGPT génération → Plan structuré

### 5. Analyse de reproductibilité
**Comment ça fonctionne** : Analyse statistique des expériences similaires pour identifier les facteurs de succès/échec.

**Ce que ça donne** : Score de risque, patterns d'échec et recommandations d'amélioration.

**À quoi ça sert** : Évaluer la fiabilité d'un protocole avant de l'exécuter, réduire les échecs expérimentaux.

**Flux de données** :
- Résultats de recherche → FailureAgent → Analyse statistique → Rapport de risque

### 6. Scraping automatisé
**Comment ça fonctionne** : Scripts périodiques récupèrent des protocoles depuis protocols.io avec extraction intelligente de conditions.

**Ce que ça donne** : Base de données publique enrichie automatiquement.

**À quoi ça sert** : Maintenir une connaissance à jour sans intervention manuelle.

**Flux de données** :
- API protocols.io → Scraper → Nettoyage → Embedding → Qdrant public_science

## Technologies utilisées

- **Backend** : Python 3.10+, FastAPI, Pydantic
- **IA** : Google Gemini 1.5 Pro, Groq Llama 3, BioBERT, BioGPT-Large
- **Base de données vectorielle** : Qdrant (Cloud + Local)
- **Embeddings** : SentenceTransformers (all-MiniLM-L6-v2, BioBERT)
- **Frontend** : React 18, Vite, Lucide Icons
- **Scraping** : BeautifulSoup, Requests, httpx
- **Workflows** : Apache Airflow
- **Sécurité** : JWT, bcrypt, rate limiting, audit logging
- **Déploiement** : Docker, Docker Compose

## Installation et configuration

### Prérequis
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- Qdrant (local ou cloud)

### Configuration
1. Cloner le repository
2. Créer un environnement virtuel Python
3. Installer les dépendances : `pip install -r requirements.txt`
4. Configurer les variables d'environnement (.env)
5. Démarrer Qdrant : `docker run -d -p 6333:6333 qdrant/qdrant`
6. Lancer le backend : `uvicorn backend.main:app --reload`
7. Installer les dépendances frontend : `cd frontend && npm install`
8. Démarrer le frontend : `npm run dev`

### Variables d'environnement
- `QDRANT_CLOUD_URL` : URL du cluster Qdrant Cloud
- `QDRANT_CLOUD_API_KEY` : Clé API Qdrant
- `GEMINI_API_KEY` : Clé API Google Gemini
- `SECRET_KEY` : Clé secrète pour JWT

## API Endpoints

### Authentification
- `POST /api/v1/auth/login` : Connexion utilisateur
- `POST /api/v1/auth/register` : Inscription

### Expériences
- `POST /api/v1/experiments/upload` : Télécharger une expérience
- `GET /api/v1/experiments/stats` : Statistiques du système

### Recherche
- `POST /api/v1/search/multimodal` : Recherche multimodale
- `POST /api/v1/search/agentic-planning` : Planification agentique
- `GET /api/v1/search/facets/{field}` : Facettes pour filtres

### Conception
- `POST /api/v1/design/variants` : Générer des variantes

### Santé
- `GET /api/v1/health` : État du système

## Sécurité et conformité

- Authentification JWT avec expiration
- Chiffrement des données sensibles
- Rate limiting par endpoint
- Audit logging complet
- Validation stricte des entrées
- Séparation des données publiques/privées

## Performance et optimisation

- Recherche vectorielle optimisée avec Qdrant
- Génération parallèle pour les preuves
- Chunking adaptatif basé sur la longueur du texte
