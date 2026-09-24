"""Tests de la chaîne RAG (rag/chain.py), sans appel API."""

# --- IMPORT MODULES ----------------------------------

from datetime import date

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chain import RAG, date_arrays, unique_events, weekend_of

# --- CONSTANTES ----------------------------------

REFERENCE = date(2026, 9, 15)

# --- FONCTIONS ----------------------------------

def make_doc(uid: str, last_date: str = "2026-10-01T20:00:00+00:00", first_date: str | None = None) -> Document:
    """Document minimal avec les métadonnées utilisées par la chaîne."""
    return Document(page_content="Concert de jazz",
                    metadata={"uid": uid, "title": f"Événement {uid}",
                              "first_date": first_date or last_date, "last_date": last_date})

def make_rag(documents: list[Document], normalize: bool = False) -> RAG:
    """RAG sur un faux index (embeddings reproductibles) et un faux LLM (réponse fixée)."""
    index = FAISS.from_documents(documents, DeterministicFakeEmbedding(size=16), normalize_L2=normalize)
    return RAG(index=index, llm=FakeListChatModel(responses=["Voici un concert."]))

class FauxExtracteur:
    """Remplace la sortie structurée du LLM : rend une période, ou lève."""

    def __init__(self, situe=True, debut=None, fin=None, erreur=False):
        self.situe_dans_le_temps, self.debut, self.fin, self.erreur = situe, debut, fin, erreur

    def invoke(self, _prompt):
        if self.erreur:
            raise RuntimeError("le modèle n'a pas répondu")
        return self

# --- TESTS ----------------------------------

def test_unique_events():
    documents = [make_doc("1"), make_doc("1"), make_doc("2"), make_doc("3")]
    assert [doc.metadata["uid"] for doc in unique_events(documents, k=2)] == ["1", "2"]

def test_ask_without_api():
    rag = make_rag([make_doc("terminé", last_date="2026-06-01T20:00:00+00:00"), make_doc("à venir")])

    result = rag.ask("Un concert de jazz ?", today=REFERENCE)

    assert result["answer"] == "Voici un concert."
    assert [source["uid"] for source in result["sources"]] == ["à venir"]
    # Un faux LLM ne sait pas produire de sortie structurée : le self-query se désactive
    assert rag.extractor is None
    assert rag.period("Un concert ce week-end ?", REFERENCE) is None

def test_date_arrays_alignes_sur_faiss():
    documents = [make_doc("1", first_date="2026-10-01T00:00:00+00:00", last_date="2026-10-05T00:00:00+00:00"),
                 make_doc("2", first_date="2026-11-02T00:00:00+00:00", last_date="2026-11-03T00:00:00+00:00")]
    index = FAISS.from_documents(documents, DeterministicFakeEmbedding(size=16))

    firsts, lasts = date_arrays(index)

    # Chaque position Faiss porte les dates du document qui s'y trouve
    for position, doc_id in index.index_to_docstore_id.items():
        metadata = index.docstore.search(doc_id).metadata
        assert firsts[position] == int(metadata["first_date"][:10].replace("-", ""))
        assert lasts[position] == int(metadata["last_date"][:10].replace("-", ""))

def test_retrieve_filtre_avant_la_recherche(monkeypatch):
    documents = [make_doc("terminé", last_date="2026-06-01T20:00:00+00:00"),
                 make_doc("octobre", last_date="2026-10-15T20:00:00+00:00"),
                 make_doc("décembre", last_date="2026-12-20T20:00:00+00:00")]
    rag = make_rag(documents)

    def uids(question="Un concert ?"):
        return sorted(doc.metadata["uid"] for doc in rag.retrieve(question, REFERENCE))

    # Filtre imposé seul : les événements terminés sont écartés
    assert uids() == ["décembre", "octobre"]

    # Période demandée : seuls les événements qui la chevauchent
    monkeypatch.setattr(rag, "period", lambda *_: (20261001, 20261031))
    assert uids() == ["octobre"]

    # Période sans aucun événement : on répond sur l'ensemble plutôt que rien
    monkeypatch.setattr(rag, "period", lambda *_: (20990101, 20990102))
    assert uids() == ["décembre", "octobre"]

def test_period_extraction():
    rag = make_rag([make_doc("1")])

    # Le week-end fourni au modèle est calculé en Python, y compris un dimanche
    assert weekend_of(date(2026, 9, 20)) == (date(2026, 9, 19), date(2026, 9, 20))

    rag.extractor = FauxExtracteur(debut=date(2026, 9, 19), fin=date(2026, 9, 20))
    assert rag.period("ce week-end ?", REFERENCE) == (20260919, 20260920)

    # Borne de fin omise : la période se réduit à son premier jour
    rag.extractor = FauxExtracteur(debut=date(2026, 9, 19))
    assert rag.period("demain ?", REFERENCE) == (20260919, 20260919)

    # Question sans dimension temporelle : aucun filtre de période
    rag.extractor = FauxExtracteur(situe=False)
    assert rag.period("un concert de jazz ?", REFERENCE) is None

    # Extraction en échec : on répond quand même, sans filtrer sur la période
    rag.extractor = FauxExtracteur(erreur=True)
    assert rag.period("ce week-end ?", REFERENCE) is None

def test_recherche_sur_index_normalise():
    # L2 activée : la question doit être normalisée comme les documents
    rag = make_rag([make_doc("1"), make_doc("2")], normalize=True)
    assert len(rag.retrieve("Un concert ?", REFERENCE)) == 2
