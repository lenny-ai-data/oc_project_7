"""Tests de la chaîne RAG (rag/chain.py), sans appel API."""

# --- IMPORT MODULES ----------------------------------

from datetime import date

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chain import RAG, unique_events

# --- FONCTIONS ----------------------------------

def make_doc(uid: str, last_date: str = "2026-10-01T20:00:00+00:00") -> Document:
    """Document minimal avec les métadonnées utilisées par la chaîne."""
    return Document(page_content="Concert de jazz", metadata={"uid": uid, "title": f"Événement {uid}", "last_date": last_date})

# --- TESTS ----------------------------------

def test_unique_events():
    documents = [make_doc("1"), make_doc("1"), make_doc("2"), make_doc("3")]
    assert [doc.metadata["uid"] for doc in unique_events(documents, k=2)] == ["1", "2"]

def test_ask_without_api():
    # Faux index (embeddings aléatoires reproductibles) et faux LLM (réponse fixée)
    documents = [make_doc("terminé", last_date="2026-06-01T20:00:00+00:00"), make_doc("à venir")]
    index = FAISS.from_documents(documents, DeterministicFakeEmbedding(size=16))
    rag = RAG(index=index, llm=FakeListChatModel(responses=["Voici un concert."]))

    result = rag.ask("Un concert de jazz ?", today=date(2026, 9, 15))

    assert result["answer"] == "Voici un concert."
    assert [source["uid"] for source in result["sources"]] == ["à venir"]
