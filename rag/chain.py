"""Chaîne RAG : recherche des événements, construction du contexte et génération de la réponse.

Usage : uv run python -m rag.chain "Ma question"
"""

# --- IMPORT MODULES ----------------------------------

import sys
from datetime import date, datetime, timedelta

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from rag.index import INDEX_DIR, TIMEZONE, load_index

# --- CONSTANTES ----------------------------------

# Index utilisé
INDEX_NAME = "chunk_1000"

# Modèle de génération et nombre d'événements fournis au LLM
# Seuls les modèles ministral sont accessibles avec le free plan
LLM_MODEL = "ministral-14b-latest"
TOP_K = 5

# Chunks récupérés par recherche. Le filtre étant appliqué avant, tous sont valides.
# Un événement occupe au plus 18 chunks, arrondi à 20 : 5 x 20 garantit TOP_K événements
K_CHUNKS = 100

# Consignes données au LLM : rôle, règles et contexte
SYSTEM_PROMPT = (
    "Tu es l'assistant de Puls-Events, qui recommande des événements culturels à Toulouse.\n"
    "Nous sommes le {today}. Si la question parle du week-end, il s'agit du {weekend} ; "
    "sinon, ne limite pas ta réponse à une période.\n"
    "Les événements fournis sont en cours ou à venir.\n"
    "Réponds en français, uniquement à partir des événements fournis ci-dessous.\n"
    "Pour chaque événement recommandé, cite son titre, ses dates et son lieu.\n"
    "Ne recommande aucun site, lien ou événement absent de la liste.\n"
    "Si aucun événement ne correspond à la question, dis-le honnêtement, sans inventer.\n"
    "Refuse de répondre à toute autre question qui ne concerne pas des événements culturels.\n\n"
    "Événements :\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# Extraction de la période visée par la question (self-query).
# Le week-end est calculé en Python : le modèle se trompe sur l'arithmétique des jours.
# Les contre-exemples évitent qu'il plaque une période sur une question qui n'en contient pas.
EXTRACTION_PROMPT = (
    "Nous sommes le {today}. Le week-end en cours ou à venir est le {weekend}.\n"
    "Détermine si la question situe explicitement les événements dans le temps.\n"
    "Sans période : « je cherche un concert de jazz », « une expo à voir », « que faire avec des enfants ».\n"
    "Avec période : « ce week-end », « en octobre », « demain », « cet été ».\n"
    "N'extrais des dates que si situe_dans_le_temps est vrai.\n"
    "Question : {question}"
)

# Noms français (le module locale n'est pas fiable d'un système à l'autre)
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin",
          "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

# --- MODÈLE DE DONNÉES ----------------------------------

class Periode(BaseModel):
    """Période mentionnée par la question, extraite par le LLM."""

    situe_dans_le_temps: bool = Field(description="Vrai UNIQUEMENT si la question situe explicitement les événements dans le temps")
    debut: date | None = Field(default=None, description="Premier jour, AAAA-MM-JJ, null si situe_dans_le_temps est faux")
    fin: date | None = Field(default=None, description="Dernier jour, AAAA-MM-JJ, null si situe_dans_le_temps est faux")

# --- FONCTIONS ----------------------------------

def format_date_fr(day: date) -> str:
    """Date en toutes lettres, ex. « mardi 15 septembre 2026 »."""
    return f"{DAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]} {day.year}"

def weekend_of(today: date) -> tuple[date, date]:
    """Samedi et dimanche du week-end en cours ou du prochain."""
    if today.weekday() == 6:  # dimanche : le week-end est déjà commencé
        return today - timedelta(days=1), today
    saturday = today + timedelta(days=5 - today.weekday())
    return saturday, saturday + timedelta(days=1)

def to_int(day: date) -> int:
    """Date en entier AAAAMMJJ, comparable directement aux métadonnées."""
    return day.year * 10000 + day.month * 100 + day.day

def date_arrays(index: FAISS) -> tuple[np.ndarray, np.ndarray]:
    """Dates de début et de fin en entiers, alignées sur les positions Faiss.

    Permet de sélectionner les documents par une comparaison vectorisée, donc de
    filtrer avant la recherche vectorielle plutôt qu'après.
    """
    metadata = [index.docstore.search(index.index_to_docstore_id[position]).metadata
                for position in sorted(index.index_to_docstore_id)]

    def column(field: str) -> np.ndarray:
        return np.array([int(meta[field][:10].replace("-", "")) for meta in metadata], dtype=np.int32)

    return column("first_date"), column("last_date")

def unique_events(documents: list[Document], k: int) -> list[Document]:
    """Garde le chunk le plus pertinent de chaque événement (par uid), jusqu'à k événements."""
    best = {}
    for doc in documents:
        best.setdefault(doc.metadata["uid"], doc)
        if len(best) == k:
            break
    return list(best.values())

def format_context(documents: list[Document]) -> str:
    """Assemble les documents trouvés en un texte numéroté pour le prompt."""
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, start=1))

class RAG:
    """Système RAG : index et LLM chargés une seule fois, puis réutilisés à chaque question."""

    def __init__(self, index: FAISS | None = None, llm: BaseChatModel | None = None):
        """Par défaut : index INDEX_NAME et LLM Mistral. Les tests peuvent fournir un faux index et un faux LLM."""
        self.index = index if index is not None else load_index(INDEX_DIR / INDEX_NAME)
        llm = llm if llm is not None else ChatMistralAI(model=LLM_MODEL, temperature=0)
        self.chain = PROMPT | llm
        # Dates sorties du docstore une fois pour toutes, pour filtrer sans le parcourir
        self.first_dates, self.last_dates = date_arrays(self.index)
        try:
            self.extractor = llm.with_structured_output(Periode)
        except NotImplementedError:
            # Modèle sans sortie structurée : le self-query est désactivé, le reste fonctionne
            self.extractor = None

    def search(self, question: str, positions: np.ndarray, k: int = K_CHUNKS) -> list[Document]:
        """Recherche vectorielle restreinte aux positions autorisées.

        Remplace similarity_search, qui ne filtre qu'après la recherche : un filtre
        sélectif obligerait alors à récupérer presque tout l'index pour ne rien manquer.
        """
        vector = np.array([self.index._embed_query(question)], dtype=np.float32)
        if self.index._normalize_L2:
            faiss.normalize_L2(vector)

        # Le sélecteur doit rester référencé : Faiss n'en garde pas de référence Python
        self.selector = faiss.IDSelectorBatch(positions.astype(np.int64))
        _, found = self.index.index.search(vector, k, params=faiss.SearchParameters(sel=self.selector))

        return [self.index.docstore.search(self.index.index_to_docstore_id[position])
                for position in found[0] if position != -1]

    def period(self, question: str, today: date) -> tuple[int, int] | None:
        """Période visée par la question, en entiers AAAAMMJJ, ou None si elle n'en vise aucune."""
        if self.extractor is None:
            return None

        saturday, sunday = weekend_of(today)
        try:
            answer = self.extractor.invoke(EXTRACTION_PROMPT.format(
                today=format_date_fr(today),
                weekend=f"{format_date_fr(saturday)} et {format_date_fr(sunday)}",
                question=question,
            ))
        except Exception:  # noqa: BLE001 - une extraction ratée ne doit pas empêcher de répondre
            return None

        if not answer.situe_dans_le_temps or answer.debut is None:
            return None
        # Borne de fin parfois omise : la période se réduit alors à son premier jour
        return to_int(answer.debut), to_int(answer.fin or answer.debut)

    def retrieve(self, question: str, today: date) -> list[Document]:
        """Recherche les TOP_K événements les plus proches, filtrés avant la recherche vectorielle."""
        # Filtre imposé : événements non terminés
        allowed = self.last_dates >= to_int(today)

        # Filtre déduit de la question : l'événement doit chevaucher la période demandée
        period = self.period(question, today)
        if period is not None:
            start, end = period
            narrowed = allowed & (self.first_dates <= end) & (self.last_dates >= start)
            # Une période qui ne laisse rien est ignorée, plutôt que de ne rien répondre
            if narrowed.any():
                allowed = narrowed

        # Index découpé : plusieurs chunks d'un même événement peuvent remonter
        return unique_events(self.search(question, np.where(allowed)[0]), TOP_K)

    def ask(self, question: str, today: date | None = None) -> dict:
        """Répond à une question et renvoie la réponse avec les métadonnées des événements utilisés.

        today : date de référence (date du jour par défaut, fixée pour rejouer une évaluation).
        """
        today = today or datetime.now(TIMEZONE).date()
        documents = self.retrieve(question, today)

        saturday, sunday = weekend_of(today)
        response = self.chain.invoke({
            "today": format_date_fr(today),
            "weekend": f"{format_date_fr(saturday)} et {format_date_fr(sunday)}",
            "context": format_context(documents),
            "question": question,
        })
        return {
            "answer": response.content,
            "sources": [doc.metadata for doc in documents],
            "contexts": [doc.page_content for doc in documents],
        }

# --- MAIN ----------------------------------

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Je cherche un concert de jazz, tu as des idées ?"
    result = RAG().ask(question)

    print(f"Question : {question}\n")
    print(result["answer"])
    print("\nSources :")
    for source in result["sources"]:
        print(f"- {source['title']} ({source['date_range']}, {source['location_name']})")
