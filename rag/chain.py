"""Chaîne RAG : recherche des événements, construction du contexte et génération de la réponse.

Usage : uv run python -m rag.chain "Ma question"
"""

# --- IMPORT MODULES ----------------------------------

import sys
from datetime import date, timedelta

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from rag.index import INDEX_DIR, load_index

# --- CONSTANTES ----------------------------------

# Index utilisé
INDEX_NAME = "no_chunk"

# Modèle de génération et nombre d'événements fournis au LLM
# Seuls les modèles ministral sont accessibles avec le free plan
LLM_MODEL = "ministral-14b-latest"
TOP_K = 5

# Candidats récupérés avant le filtre sur les dates (≈ 1 événement sur 4 est à venir)
FETCH_K = 200

# Prompt : rôle, règles, contexte et question
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es l'assistant de Puls-Events, qui recommande des événements culturels à Toulouse.\n"
     "Nous sommes le {today} ; « ce week-end » désigne le {weekend}.\n"
     "Les événements fournis sont en cours ou à venir.\n"
     "Réponds en français, uniquement à partir des événements fournis ci-dessous.\n"
     "Pour chaque événement recommandé, cite son titre, ses dates et son lieu.\n"
     "Ne recommande aucun site, lien ou événement absent de la liste.\n"
     "Si aucun événement ne correspond à la question, dis-le honnêtement, sans inventer.\n"
     "Refuse de répondre à toute autre question qui ne concerne pas des événements culturels.\n\n"
     "Événements :\n{context}"),
    ("human", "{question}"),
])

# Noms français (le module locale n'est pas fiable d'un système à l'autre)
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin",
          "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

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

def format_context(documents: list[Document]) -> str:
    """Assemble les documents trouvés en un texte numéroté pour le prompt."""
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, start=1))

class RAG:
    """Système RAG : index et LLM chargés une seule fois, puis réutilisés à chaque question."""

    def __init__(self, index_name: str = INDEX_NAME):
        self.index = load_index(INDEX_DIR / index_name)
        self.chain = PROMPT | ChatMistralAI(model=LLM_MODEL, temperature=0)

    def ask(self, question: str, today: date | None = None) -> dict:
        """Répond à une question et renvoie la réponse avec les métadonnées des événements utilisés.

        today : date de référence (date du jour par défaut, fixée pour rejouer une évaluation).
        """
        today = today or date.today()

        # Recherche vectorielle puis filtre : on ne garde que les événements non terminés
        documents = self.index.similarity_search(
            question, k=TOP_K, fetch_k=FETCH_K,
            filter=lambda metadata: metadata["last_date"][:10] >= today.isoformat(),
        )

        saturday, sunday = weekend_of(today)
        response = self.chain.invoke({
            "today": format_date_fr(today),
            "weekend": f"{format_date_fr(saturday)} et {format_date_fr(sunday)}",
            "context": format_context(documents),
            "question": question,
        })
        return {"answer": response.content, "sources": [doc.metadata for doc in documents]}

# --- MAIN ----------------------------------

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Je cherche un concert de jazz, tu as des idées ?"
    result = RAG().ask(question)

    print(f"Question : {question}\n")
    print(result["answer"])
    print("\nSources :")
    for source in result["sources"]:
        print(f"- {source['title']} ({source['date_range']}, {source['location_name']})")
