# Développement d'un assistant pour la recommandation d'évènements culturels

POC d'un chatbot pour **Puls-Events** : il répond aux questions des utilisateurs sur les événements culturels à partir des données **Open Agenda**, grâce à un système **RAG** (recherche vectorielle + génération de réponse).

Stack : **LangChain**, **Mistral** (embeddings et LLM), **Faiss** (base vectorielle).

📄 Choix techniques, données et résultats : voir le [rapport technique](docs/rapport_technique.md).

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

## Construction de l'index vectoriel

Prérequis : une clé API Mistral dans un fichier `.env` à la racine (`MISTRAL_API_KEY=...`), à vérifier avec `scripts/check_env.py`.

```bash
# Vectorisation (mistral-embed) et index Faiss -> data/index/
uv run python -m rag.index
# Ou une seule configuration
uv run python -m rag.index chunk_1000
```

Deux index sont construits, sans découpage (`no_chunk`, ~1 min) et avec découpage à 1 000 caractères (`chunk_1000`, ~2 min).

## Poser une question au système RAG

```bash
# Recherche des événements à venir les plus proches + réponse générée par Mistral
uv run python -m rag.chain "Je cherche un concert de jazz, tu as des idées ?"
```

## Évaluation

Le jeu de test annoté (20 questions, date de référence fixée) est dans `eval/test_set.json`.

```bash
# 1. Pose les questions au RAG, calcule le hit@5 et sauvegarde les réponses -> eval/results/<index>.json
uv run python -m rag.evaluate run no_chunk
# 2. Ajoute les scores Ragas (juge ministral-8b) à partir des réponses sauvegardées
uv run python -m rag.evaluate ragas no_chunk
```

> Ragas 0.4.3 ne s'importe pas tel quel avec `langchain-community` 0.4 ([issue #2745](https://github.com/vibrantlabsai/ragas/issues/2745)). `rag/evaluate.py` applique un contournement avant l'import : utiliser Ragas via ce module plutôt qu'un `import ragas` direct.

## Tests

```bash
uv run pytest
```
