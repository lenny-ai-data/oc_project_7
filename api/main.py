"""API FastAPI qui expose le système RAG aux équipes métier.

Usage : uv run uvicorn api.main:app --reload   (documentation interactive sur /docs)
"""

# --- IMPORT MODULES ----------------------------------

import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from rag.chain import INDEX_NAME, LLM_MODEL, RAG, TIMEZONE, TOP_K
from rag.collect import CITY, START_DATE
from rag.index import EMBEDDING_MODEL, INDEX_DIR, rebuild_index

# --- CONSTANTES ----------------------------------

# MISTRAL_API_KEY et AUTH_TOKEN
load_dotenv()

# Fichier de vecteurs, dont la date de modification sert de date de construction de l'index
INDEX_FILE = INDEX_DIR / INDEX_NAME / "index.faiss"

# En-tête qui porte le jeton (bouton « Authorize » dans Swagger)
TOKEN_HEADER = APIKeyHeader(name="X-Token", auto_error=False, description="Valeur de AUTH_TOKEN")

# Une seule reconstruction à la fois
REBUILD_LOCK = threading.Lock()

# Commit déployé, injecté par Render ; absent en local
REVISION = os.getenv("RENDER_GIT_COMMIT")

# --- ÉTAT DE L'API ----------------------------------

# Système RAG chargé une seule fois au démarrage (voir lifespan)
# Les tests remplacent rag_system par un RAG à faux index et faux LLM
rag_system: RAG | None = None
startup_error: str | None = None

# --- MODÈLES DE DONNÉES ----------------------------------

class Health(BaseModel):
    """État de l'API et de l'index."""

    status: str = Field(description="ok si l'API peut répondre, degraded si l'index est absent")
    index: str
    revision: str | None = Field(default=None, description="Commit déployé, pour vérifier qu'une mise en ligne est effective")
    error: str | None = None

class Metadata(BaseModel):
    """Périmètre des données et configuration du POC."""

    city: str
    period_start: str = Field(description="Les événements terminés avant cette date sont exclus de la collecte")
    events: int = Field(description="Nombre d'événements distincts indexés")
    vectors: int = Field(description="Nombre de vecteurs (chunks)")
    dimension: int
    embedding_model: str
    llm_model: str
    top_k: int = Field(description="Nombre d'événements fournis au LLM pour construire la réponse")
    index_name: str
    index_built_at: str

class Question(BaseModel):
    """Question posée au système RAG."""

    # str_strip_whitespace : rejet des questions vides
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=3, max_length=500,
        description="Question en langage naturel sur les événements culturels de Toulouse",
        examples=["Je cherche un spectacle pour enfant ce week-end, tu as des idées ?"],
    )
    today: date | None = Field(
        default=None,
        description="Date de référence pour « ce week-end » et le filtre des événements passés (aujourd'hui par défaut)",
    )

class Source(BaseModel):
    """Sources utilisées par le LLM, réduit aux métadonnées utiles."""

    uid: str
    title: str
    date_range: str
    location_name: str | None = None
    url: str | None = None

class Answer(BaseModel):
    """Réponse générée et événements sur lesquels elle s'appuie."""

    answer: str
    sources: list[Source]

class Rebuild(BaseModel):
    """Résultat d'une reconstruction de l'index."""

    events: int
    vectors: int
    seconds: float = Field(description="Durée de la reconstruction")

# --- FONCTIONS ----------------------------------

def require_rag() -> RAG:
    """Renvoie le système RAG chargé, ou erreur 503"""
    if rag_system is None:
        raise HTTPException(503, f"Index indisponible : {startup_error}")
    return rag_system

def check_token(token: str | None) -> None:
    """Vérifie le jeton, ou désactive la route si AUTH_TOKEN n'est pas défini"""
    expected = os.getenv("AUTH_TOKEN")
    if not expected:
        raise HTTPException(503, "AUTH_TOKEN n'est pas défini, route désactivée")
    # compare_digest plutôt que != : comparaison à durée constante pour ne rien révéler
    if not secrets.compare_digest(token or "", expected):
        raise HTTPException(401, "Jeton invalide")

def count_events(rag: RAG) -> int:
    """Compte les événements distincts de l'index"""
    docstore = rag.index.docstore
    return len({docstore.search(doc_id).metadata["uid"] for doc_id in rag.index.index_to_docstore_id.values()})

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge l'index et le client Mistral au démarrage.

    Un index absent ne fait pas échouer le démarrage : l'API signale le problème sur /health
    """
    global rag_system, startup_error
    try:
        rag_system = RAG()
    except Exception as error:  # noqa: BLE001 - toute panne au démarrage doit être signalée, pas propagée
        startup_error = f"{type(error).__name__} : {error}"
    yield

# --- API ----------------------------------

app = FastAPI(
    title="Puls-Events - API",
    description="📅 Répond à vos questions sur les événements culturels de Toulouse, à partir des données Open Agenda.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", summary="État de l'API et de l'index")
def health() -> Health:
    """Vérifie que l'API est prête à répondre (index chargé)"""
    if rag_system is None:
        return Health(status="degraded", index="absent", revision=REVISION, error=startup_error)
    return Health(status="ok", index=INDEX_NAME, revision=REVISION)

@app.get("/metadata", summary="Périmètre et configuration du POC")
def metadata() -> Metadata:
    """Décrit les données couvertes et les modèles utilisés."""
    rag = require_rag()
    return Metadata(
        city=CITY,
        period_start=START_DATE,
        events=count_events(rag),
        vectors=rag.index.index.ntotal,
        dimension=rag.index.index.d,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        top_k=TOP_K,
        index_name=INDEX_NAME,
        index_built_at=datetime.fromtimestamp(INDEX_FILE.stat().st_mtime, tz=TIMEZONE).date().isoformat(),
    )

@app.post("/ask", summary="Poser une question sur les événements")
def ask(request: Question, token: str | None = Security(TOKEN_HEADER)) -> Answer:
    """Recherche les événements à venir les plus proches de la question et génère une réponse.

    Protégée par un jeton : chaque question consomme du quota Mistral.
    """
    check_token(token)
    rag = require_rag()
    try:
        result = rag.ask(request.question, request.today)
    except Exception as error:
        # Mistral indisponible, quota dépassé, clé invalide : l'API dépend d'un service tiers
        raise HTTPException(502, f"Le service Mistral n'a pas répondu : {type(error).__name__}") from error
    return Answer(answer=result["answer"], sources=result["sources"])

@app.post("/rebuild", summary="Reconstruire l'index à partir des données Open Agenda")
def rebuild(token: str | None = Security(TOKEN_HEADER)) -> Rebuild:
    """Recollecte, nettoie et revectorise les événements, puis recharge l'index sans redémarrer l'API.

    Opération longue (plusieurs minutes) et facturée par Mistral, protégée par un jeton.
    L'index en service n'est remplacé qu'en cas de succès.
    """
    global rag_system, startup_error
    check_token(token)
    if not REBUILD_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Une reconstruction est déjà en cours")
    try:
        start = time.perf_counter()
        index = rebuild_index(INDEX_NAME)
        rag_system, startup_error = RAG(index=index), None
        return Rebuild(events=count_events(rag_system), vectors=index.index.ntotal,
                       seconds=round(time.perf_counter() - start, 1))
    except Exception as error:
        # Open Agenda injoignable, données illisibles, quota Mistral dépassé
        raise HTTPException(502, f"Échec de la reconstruction : {type(error).__name__}") from error
    finally:
        REBUILD_LOCK.release()
