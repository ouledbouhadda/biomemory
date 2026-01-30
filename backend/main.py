from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
from backend.config.settings import get_settings
from backend.api.routes import auth, experiments, search, design, health
from backend.security.rate_limiting import RateLimiter, RateLimitMiddleware
from backend.security.audit_logger import AuditLogger, AuditMiddleware
from backend.services.qdrant_service import get_qdrant_service
from backend.database.user_repository import UserRepository

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("biomemory")

settings = get_settings()
if settings.DEBUG:
    logging.getLogger("biomemory").setLevel(logging.DEBUG)

_user_repository = None
def get_user_repository() -> UserRepository:
    return _user_repository
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _user_repository
    logger.info("Starting BioMemory API v%s (debug=%s)", settings.APP_VERSION, settings.DEBUG)
    try:
        qdrant = get_qdrant_service()
        await qdrant.init_collections()
        logger.info("Qdrant collections initialized")
        _user_repository = UserRepository(qdrant.private_client)
        logger.info("User repository initialized")
    except Exception as e:
        logger.warning("Qdrant initialization warning: %s", e)
    logger.info("BioMemory API ready! Docs: http://localhost:8000/api/docs")
    yield
    logger.info("Shutting down BioMemory API...")
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multimodal Biological Design & Discovery Intelligence System",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.biomemory.dev"]
    )
rate_limiter = RateLimiter(max_requests=settings.RATE_LIMIT_PER_MINUTE)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
audit_logger = AuditLogger()
app.add_middleware(AuditMiddleware, audit_logger=audit_logger)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "error_type": "ValueError"
        }
    )
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    if settings.DEBUG:
        raise exc
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__
        }
    )
app.include_router(
    health.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Health"]
)
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"]
)
app.include_router(
    experiments.router,
    prefix=f"{settings.API_V1_PREFIX}/experiments",
    tags=["Experiments"]
)
app.include_router(
    search.router,
    prefix=f"{settings.API_V1_PREFIX}/search",
    tags=["Search"]
)
app.include_router(
    design.router,
    prefix=f"{settings.API_V1_PREFIX}/design",
    tags=["Design"]
)
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs",
        "health": "/health"
    }

@app.post("/agentic-test")
async def agentic_test(query: str):
    """Test endpoint for agentic planning"""
    import random
    
    # Simuler différents scénarios selon la requête
    num_experiments = random.randint(8, 28)
    success_rate = random.randint(65, 95)
    
    # Générer des recommandations contextualisées
    if "PCR" in query.upper():
        conditions = "température optimale entre 35°C et 40°C"
        steps = "1. Préparer le mélange réactionnel avec 0.5 µM de chaque primer\n2. Utiliser 0.5 U/µL de Taq polymerase\n3. Effectuer 30-35 cycles d'amplification\n4. Vérifier par électrophorèse en gel d'agarose"
        scenario_text = f"En se basant sur {num_experiments} expériences similaires, votre PCR a {success_rate}% de chances de réussite aux conditions et étapes suivantes: {conditions}. Suivez ces étapes: {steps}"
    elif "culture" in query.lower() or "cellulaire" in query.lower():
        conditions = "milieu DMEM à 37°C avec 5% CO2 et 10-15% de sérum"
        steps = "1. Ensemencer à une densité de 1-2×10⁵ cellules/ml\n2. Maintenir l'incubation à 37°C, 5% CO2\n3. Effectuer un changement de milieu tous les 2-3 jours\n4. Subculture tous les 4-5 jours pour éviter la confluence"
        scenario_text = f"En se basant sur {num_experiments} expériences similaires, votre culture cellulaire réussira avec {success_rate}% de probabilité sous les conditions suivantes: {conditions}. Suivez ce protocole: {steps}"
    elif "ADN" in query.upper() or "extraction" in query.lower():
        conditions = "méthode phénol-chloroforme classique ou colonne de purification"
        steps = "1. Lyser les tissus en tampon de lyse\n2. Digérer les protéines avec protéinase K\n3. Extraire avec phénol/chloroforme/alcool isoamylique\n4. Précipiter l'ADN à l'éthanol 70%\n5. Réhydrater dans du tampon TE"
        scenario_text = f"En se basant sur {num_experiments} expériences similaires de biologie moléculaire, votre extraction d'ADN a {success_rate}% de chances de réussite. Utilisez {conditions}. Procédez comme suit: {steps}"
    else:
        scenario_text = f"En se basant sur {num_experiments} expériences similaires trouvées, votre expérience devrait réussir avec une probabilité de {success_rate}%. Les étapes clés sont: 1. Préparer correctement les réactifs, 2. Respecter les conditions de température/pH, 3. Valider les résultats par une méthode complémentaire, 4. Documenter chaque étape."
    
    return {
        "status": "success",
        "query": query,
        "parsed_experiment": {
            "type_experiment": "Expérience biologique",
            "organism": "Multiple",
            "conditions": "À optimiser selon le protocole"
        },
        "recommendations": scenario_text,
        "qdrant_insights": {
            "similar_experiments_found": num_experiments,
            "success_rate_percentage": success_rate,
            "vector_search": [],
            "hybrid_search": [],
            "grouped_by_organism": {},
            "total_experiments": num_experiments
        },
        "pipeline_time_ms": random.randint(800, 1500)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )