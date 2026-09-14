# Développement d'un assistant pour la recommandation d'évènements culturels

POC d'un chatbot pour **Puls-Events** : il répond aux questions des utilisateurs sur les événements culturels à partir des données **Open Agenda**, grâce à un système **RAG** (recherche vectorielle + génération de réponse).

Stack : **LangChain**, **Mistral** (embeddings et LLM), **Faiss** (base vectorielle).

## Installation

Prérequis : `uv`.

```bash
# Récupération du projet
git clone https://github.com/lenny-ai-data/oc_project_7.git
cd oc_project_7
# Création de l'environnement virtuel
uv sync
```

## Vérification de l'environnement

```bash
uv run python scripts/check_env.py
```

Le script réalise un test d'import des briques principales (Faiss, LangChain, Mistral).

## Préparation des données

```bash
# Collecte des événements Open Agenda de Toulouse -> data/raw/
uv run python -m rag.collect
# Nettoyage -> data/processed/
uv run python -m rag.preprocess
# Aperçu des documents à vectoriser
uv run python -m rag.documents
```

L'analyse exploratoire qui justifie ces choix est dans `scripts/eda_openagenda.ipynb`.

## Tests

```bash
uv run pytest
```
