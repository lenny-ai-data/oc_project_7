"""Chaîne RAG : recherche des événements, construction du contexte et génération de la réponse.

Usage : uv run python -m rag.chain "Ma question"
"""

# --- IMPORT MODULES ----------------------------------

import sys

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

# Prompt : rôle, règles, contexte et question
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es l'assistant de Puls-Events, qui recommande des événements culturels à Toulouse.\n"
     "Réponds en français, uniquement à partir des événements fournis ci-dessous.\n"
     "Pour chaque événement recommandé, cite son titre, ses dates et son lieu.\n"
     "Si aucun événement ne correspond à la question, dis-le honnêtement, sans inventer.\n"
     "Refuse de répondre à toute autre question qui ne concerne pas des événements culturels.\n\n"
     "Événements :\n{context}"),
    ("human", "{question}"),
])

# --- FONCTIONS ----------------------------------

def format_context(documents: list[Document]) -> str:
    """Assemble les documents trouvés en un texte numéroté pour le prompt."""
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(documents, start=1))

class RAG:
    """Système RAG : index et LLM chargés une seule fois, puis réutilisés à chaque question."""

    def __init__(self, index_name: str = INDEX_NAME):
        self.index = load_index(INDEX_DIR / index_name)
        self.chain = PROMPT | ChatMistralAI(model=LLM_MODEL, temperature=0)

    def ask(self, question: str) -> dict:
        """Répond à une question et renvoie la réponse avec les métadonnées des événements utilisés."""
        documents = self.index.similarity_search(question, k=TOP_K)
        response = self.chain.invoke({"context": format_context(documents), "question": question})
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
