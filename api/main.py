"""API FastAPI qui expose le système RAG aux équipes métier.

Usage : uv run uvicorn api.main:app --reload   (documentation interactive sur /docs)
"""

# --- IMPORT MODULES ----------------------------------

from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rag.chain import INDEX_NAME, LLM_MODEL, RAG, TOP_K
from rag.collect import CITY, START_DATE
from rag.index import EMBEDDING_MODEL, INDEX_DIR

# --- CONSTANTES ----------------------------------

# Fichier de vecteurs, dont la date de modification sert de date de construction de l'index
INDEX_FILE = INDEX_DIR / INDEX_NAME / "index.faiss"

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
        examples=["Je cherche un concert de jazz, tu as des idées ?"],
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

# --- FONCTIONS ----------------------------------

def require_rag() -> RAG:
    """Renvoie le système RAG chargé, ou erreur 503"""
    if rag_system is None:
        raise HTTPException(503, f"Index indisponible : {startup_error}")
    return rag_system

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
    except Exception as error:
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
        return Health(status="degraded", index="absent", error=startup_error)
    return Health(status="ok", index=INDEX_NAME)

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
        index_built_at=datetime.fromtimestamp(INDEX_FILE.stat().st_mtime).date().isoformat(),
    )

@app.post("/ask", summary="Poser une question sur les événements")
def ask(request: Question) -> Answer:
    """Recherche les événements à venir les plus proches de la question et génère une réponse."""
    rag = require_rag()
    try:
        result = rag.ask(request.question, request.today)
    except Exception as error:
        # Mistral indisponible, quota dépassé, clé invalide : l'API dépend d'un service tiers
        raise HTTPException(502, f"Le service Mistral n'a pas répondu : {type(error).__name__}") from error
    return Answer(answer=result["answer"], sources=result["sources"])
