"""Tests de l'API (api/main.py), sans appel réseau : faux index, faux LLM, reconstruction simulée."""

# --- IMPORT MODULES ----------------------------------

import pytest
from fastapi.testclient import TestClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from api import main
from rag.chain import RAG

# --- CONSTANTES ----------------------------------

TOKEN = "jeton-de-test"
HEADERS = {"X-Token": TOKEN}
QUESTION = {"question": "Je cherche un concert de jazz ?"}

# --- FIXTURES ----------------------------------

def make_doc(uid: str, title: str) -> Document:
    """Événement à venir, avec des métadonnées en trop pour vérifier le filtrage de la réponse."""
    return Document(
        page_content=f"Titre : {title}",
        metadata={"uid": uid, "title": title, "date_range": "Mardi 1 décembre 2026, 20h00",
                  "last_date": "2026-12-01T20:00:00+00:00", "location_name": "Le Bikini",
                  "url": "https://exemple.fr", "status": "Prévu", "agenda": "Test"},
    )

@pytest.fixture
def fake_rag() -> RAG:
    """Vrai système RAG branché sur un faux index et un faux LLM : aucun appel à Mistral."""
    documents = [make_doc("1", "Concert de jazz"), make_doc("2", "Exposition")]
    index = FAISS.from_documents(documents, DeterministicFakeEmbedding(size=16))
    return RAG(index=index, llm=FakeListChatModel(responses=["Voici un concert."]))

@pytest.fixture
def client(fake_rag, monkeypatch) -> TestClient:
    """Client de test avec un index en mémoire et un jeton connu.

    Sans gestionnaire de contexte, le lifespan n'est pas lancé : le vrai index n'est jamais chargé.
    monkeypatch remet l'état du module en place après chaque test.
    """
    monkeypatch.setenv("AUTH_TOKEN", TOKEN)
    monkeypatch.setattr(main, "rag_system", fake_rag)
    monkeypatch.setattr(main, "startup_error", None)
    return TestClient(main.app)

# --- TESTS ----------------------------------

def test_diagnostic(client, monkeypatch):
    # revision : renseignée par l'hébergeur, absente en local
    assert client.get("/health").json() == {"status": "ok", "index": main.INDEX_NAME, "revision": None, "error": None}
    data = client.get("/metadata").json()
    assert data["city"] == "Toulouse"
    assert (data["events"], data["vectors"]) == (2, 2)

    # Index perdu : l'API reste joignable et le signale
    monkeypatch.setattr(main, "rag_system", None)
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/metadata").status_code == 503
    assert client.post("/ask", json=QUESTION, headers=HEADERS).status_code == 503

def test_ask(client):
    data = client.post("/ask", json=QUESTION, headers=HEADERS).json()
    assert data["answer"] == "Voici un concert."
    assert len(data["sources"]) == 2
    # La réponse ne contient que les métadonnées utiles au client (ni status, ni agenda)
    assert set(data["sources"][0]) == {"uid", "title", "date_range", "location_name", "url"}

# "  " couvre aussi la question vide, rabotée avant la validation de longueur
@pytest.mark.parametrize("question", ["  ", "a" * 501])
def test_ask_question_invalide(client, question):
    assert client.post("/ask", json={"question": question}, headers=HEADERS).status_code == 422

def test_authentification(client, monkeypatch):
    assert client.post("/ask", json=QUESTION).status_code == 401
    assert client.post("/ask", json=QUESTION, headers={"X-Token": "faux"}).status_code == 401

    # Variable absente : les deux routes payantes sont désactivées
    monkeypatch.delenv("AUTH_TOKEN")
    assert client.post("/ask", json=QUESTION, headers=HEADERS).status_code == 503
    assert client.post("/rebuild", headers=HEADERS).status_code == 503

def test_ask_erreur_mistral(client, fake_rag, monkeypatch):
    def echec(*args, **kwargs):
        raise TimeoutError("Mistral injoignable")

    monkeypatch.setattr(fake_rag, "ask", echec)
    assert client.post("/ask", json=QUESTION, headers=HEADERS).status_code == 502

def test_rebuild(client, fake_rag, monkeypatch):
    # Reconstruction simulée : la vraie prend plusieurs minutes et est facturée
    # Clé Mistral nécessaire pour le RAG.__init__, mais ChatMistralAI n'appelle rien à la création
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-de-test")  
    monkeypatch.setattr(main, "rebuild_index", lambda name: fake_rag.index)
    # RAG a None pour verifier si le rebuild le reconstruit bien
    monkeypatch.setattr(main, "rag_system", None)
    data = client.post("/rebuild", headers=HEADERS).json()

    assert (data["events"], data["vectors"]) == (2, 2)
    assert main.rag_system is not None

    # Test du verrou
    main.REBUILD_LOCK.acquire()
    try:
        assert client.post("/rebuild", headers=HEADERS).status_code == 409
    finally:
        main.REBUILD_LOCK.release()

def test_rebuild_echec(client, fake_rag, monkeypatch):
    def echec(name):
        raise ConnectionError("Open Agenda injoignable")

    monkeypatch.setattr(main, "rebuild_index", echec)

    assert client.post("/rebuild", headers=HEADERS).status_code == 502
    assert main.rag_system is fake_rag  # l'index en service n'a pas été remplacé
