# Développement d'un assistant pour la recommandation d'évènements culturels

POC d'un chatbot pour **Puls-Events** : il répond aux questions des utilisateurs sur les événements culturels à partir des données **Open Agenda**, grâce à un système **RAG**.

> 💢 **Instances de démonstration en ligne** 💢 (compter ~1 min de réveil pour chaque instance) :
> - 🔌API : <https://puls-events-api.onrender.com/docs>
> - 💬 Interface de chat : <https://puls-events-chat.onrender.com> (identifiants communiqués séparément)

Stack : système **LangChain**, base vectorielle **Faiss** et modèles **Mistral** (embeddings et LLM).

📄 Choix techniques, données, déploiement et résultats tracés dans le [rapport technique](docs/rapport_technique.md).

## Arborescence

```
P7/
├── rag/                      # Logique métier
│   ├── collect.py            # Collecte des événements Open Agenda -> data/raw/
│   ├── preprocess.py         # Nettoyage des événements -> data/processed/
│   ├── documents.py          # Construction des Documents LangChain et découpage en chunks
│   ├── index.py              # Vectorisation Mistral et index Faiss -> data/index/
│   ├── chain.py              # Chaîne RAG : recherche, prompt et génération
│   └── evaluate.py           # Évaluation : exécution du jeu de test, hit@5, scores Ragas
├── api/
│   └── main.py               # API FastAPI : /health, /metadata, /ask, /rebuild
├── scripts/
│   ├── check_env.py          # Vérification des imports et de la clé API
│   ├── ask_api.py            # Client en ligne de commande de l'API
│   ├── chat_ui.py            # Interface de chat Gradio, cliente de l'API
│   ├── benchmark_faiss.py    # Comparaison des algorithmes d'index Faiss
│   └── eda_openagenda.ipynb  # Analyse exploratoire justifiant la collecte et le nettoyage
├── eval/                     # Jeu de test annoté, résultats et suivi des itérations
├── tests/                    # Tests unitaires (pytest)
├── docs/                     # Rapport technique
├── data/                     # Données générées, seul l'index chunk_1000 est versionné
├── .github/workflows/ci.yml  # Lint, tests, build de l'image, déploiement Render
├── run.py                    # Lancement local de l'API
├── Dockerfile                # Image de l'API, index embarqué
├── Dockerfile.ui             # Image de l'interface de chat
└── pyproject.toml / uv.lock  # Dépendances (gestionnaire uv)
```

## Installation

Prérequis : `uv`, et un `.env` rempli à partir de `.env.example`.

```bash
# Récupération du projet
git clone https://github.com/lenny-ai-data/oc_project_7.git
cd oc_project_7
# Création de l'environnement virtuel
uv sync
# Test d'import des briques principales + appel API de test
uv run python scripts/check_env.py
```

## Données et index (optionnel)

L'index `chunk_1000` est versionné : cette étape ne sert qu'à tout reconstruire.

```bash
# Collecte Open Agenda (Toulouse) > data/raw/, puis nettoyage > data/processed/
uv run python -m rag.collect
uv run python -m rag.preprocess
```

L'analyse exploratoire qui justifie ces choix est dans `scripts/eda_openagenda.ipynb`.

```bash
# Vectorisation (mistral-embed) et index Faiss > data/index/ (toutes les configurations, ou une seule)
uv run python -m rag.index
uv run python -m rag.index chunk_1000
```
Deux configurations : `no_chunk` (sans découpage, ~1 min) et `chunk_1000` (1 000 caractères, recouvrement 150, ~2 min). 

## Mise en service de l'API en local

```bash
# Mise en service via uvicorn
uv run python run.py
# Ou via Docker (index embarqué, opérationnel dès le démarrage)
docker build -t puls-events .
docker run --rm -p 8000:8000 --env-file .env puls-events
```

| Méthode | Route | Rôle | Jeton |
|---|---|---|---|
| GET | `/health` | État de l'API et de l'index | non |
| GET | `/metadata` | Zone, période, volumes et modèles utilisés | non |
| POST | `/ask` | Question → réponse et sources | oui |
| POST | `/rebuild` | Reconstruction complète de l'index (uniquement en local) | oui |

Quelques remarques :
- Le jeton `AUTH_TOKEN` s'envoie dans l'en-tête `X-Token`. 
- Le champ optionnel `today` de `/ask` (`"2026-09-15"`) fixe la date de référence, pour rejouer une démonstration à date constante.
- Chaque appel à `/ask` déclenche trois requêtes Mistral : repérage de la période visée par la question, vectorisation, puis génération de la réponse.
- La reconstruction de l'index, `/rebuild`, rejoue toute la chaîne (collecte, nettoyage, vectorisation) et recharge l'index sans redémarrer l'API. Compter 1 à 3 minutes et une consommation de quota Mistral.

## Interroger l'API

Les clients visent par défaut l'**API Render**. Pour viser l'API locale, définir `API_URL=http://localhost:8000` (dans le `.env` ou le terminal).

| Client | Commande |
|---|---|
| Documentation interactive | `<API_URL>/docs`, bouton **Authorize** pour le jeton |
| Script en ligne de commande | `uv run python scripts/ask_api.py "Je cherche une pièce de théâtre, tu as des idées ?"` |
| Interface de chat (http://127.0.0.1:7860) | `uv run python scripts/chat_ui.py`, dispose de sa propre image Docker `Dockerfile.ui` |

Ou directement via une requête :

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -H "X-Token: $AUTH_TOKEN" \
  -d '{"question": "Je cherche une pièce de théâtre, tu as des idées ?"}'
```

## Évaluation

Jeu de test annoté (20 questions, date de référence fixée) dans `eval/test_set.json`.

```bash
# Pose les questions au RAG, calcule le hit@5 et sauvegarde les réponses > eval/results/<index>.json
uv run python -m rag.evaluate run chunk_1000
# Scores Ragas (juge ministral-14b) à partir des réponses sauvegardées
uv run python -m rag.evaluate ragas chunk_1000
```

Ragas 0.4.3 ne s'importe pas tel quel avec `langchain-community` 0.4 ([issue #2745](https://github.com/vibrantlabsai/ragas/issues/2745)) : `rag/evaluate.py` applique un contournement avant l'import.

## Tests et CI/CD

```bash
# Lancement des tests
uv run pytest
# Avec rapport de couverture (rag/ et api/)
uv run pytest --cov --cov-report=html
```

À chaque push, `.github/workflows/ci.yml` lance `ruff`, `pytest` (seuil de couverture 80 %) et le build de l'image Docker. Les tests qui demandent une clé API ou des données brutes sont ignorés automatiquement. Un push sur `main` déploie ensuite l'**API sur Render**.

- Si l'instance est en sommeil elle redémarre en ~1 minute
- La route `/rebuild` dépasse la limite de mémoire allouée à l'instance, elle n'est pas fonctionnelle sur cette instance.

L'image de l'**interface de chat** est également déployée sur Render, comme un second service, hors du pipeline.
