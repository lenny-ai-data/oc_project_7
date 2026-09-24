# Développement d'un assistant pour la recommandation d'évènements culturels

POC d'un chatbot pour **Puls-Events** : il répond aux questions des utilisateurs sur les événements culturels à partir des données **Open Agenda**, grâce à un système **RAG** (recherche vectorielle + génération de réponse).

Stack : **LangChain**, base vectorielle **Faiss** et modèles **Mistral** (embeddings et LLM).

📄 Choix techniques, données et résultats : voir le [rapport technique](docs/rapport_technique.md).

⏩ **Instance de démonstration en ligne** : <https://puls-events-api.onrender.com/docs> (compter ~1 min de réveil de l'instance).

## Arborescence

```
P7/
├── rag/                      # Logique métier, réutilisée par les scripts, l'API et les tests
│   ├── collect.py            # Collecte des événements Open Agenda -> data/raw/
│   ├── preprocess.py         # Nettoyage des événements -> data/processed/
│   ├── documents.py          # Construction des Documents LangChain et découpage en chunks
│   ├── index.py              # Vectorisation Mistral et index Faiss -> data/index/
│   ├── chain.py              # Chaîne RAG : recherche, prompt et génération (classe RAG)
│   └── evaluate.py           # Évaluation : exécution du jeu de test, hit@5, scores Ragas
├── api/
│   └── main.py               # API FastAPI : /health, /metadata, /ask, /rebuild
├── scripts/
│   ├── check_env.py          # Vérification des imports et de la clé API Mistral
│   ├── benchmark_faiss.py    # Comparaison des algorithmes d'index Faiss (Flat, HNSW, IVF, PQ)
│   └── eda_openagenda.ipynb  # Analyse exploratoire justifiant la collecte et le nettoyage
├── eval/
│   ├── test_set.json         # Jeu de test annoté (20 questions)
│   ├── results/              # Réponses, sources, contextes et scores par index
│   └── iterations.md         # Suivi des itérations d'amélioration
├── tests/                    # Tests unitaires (pytest)
├── docs/
│   └── rapport_technique.md  # Ce rapport
├── data/                     # Données générées (raw/, processed/, index/), seul l'index chunk_1000 est versionné
├── .github/workflows/ci.yml  # Lint, tests, build de l'image, déploiement Render
├── Dockerfile                # Image de l'API, index embarqué
├── .dockerignore             # Contexte de build réduit au nécessaire
├── pyproject.toml / uv.lock  # Dépendances (gestionnaire uv)
└── README.md                 # Installation et commandes
```

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

Créer le `.env` ou charger les variables dans l'environnement.

```bash
# Test d'import des briques principales (Faiss, LangChain, Mistral) + appel API de test
uv run python scripts/check_env.py
```

## Préparation des données (optionnel → base vectorielle dans le dépôt)

```bash
# Collecte des événements Open Agenda de Toulouse > data/raw/
uv run python -m rag.collect
# Preprocessing > data/processed/
uv run python -m rag.preprocess
# Aperçu des documents à vectoriser (visualisation uniquement)
uv run python -m rag.documents
```

L'analyse exploratoire qui justifie ces choix est dans `scripts/eda_openagenda.ipynb`.

## Construction de l'index vectoriel (optionnel → base vectorielle dans le dépôt)

Prérequis : clé API Mistral `MISTRAL_API_KEY` dans l'environnement, à vérifier avec `scripts/check_env.py`.

Deux configurations sont prédéfinies :
- `no_chunk` (chunk_size=None, chunk_overlap=0)
- `chunk_1000` (chunk_size=1000, chunk_overlap=150)

```bash
# Vectorisation (mistral-embed) et index Faiss > data/index/, pour toutes les configurations existantes
uv run python -m rag.index
# Ou une configuration spécifique
uv run python -m rag.index chunk_1000
```
Deux index sont construits, sans découpage (`no_chunk`, ~1 min) et avec découpage à 1 000 caractères (`chunk_1000`, ~2 min).

## Poser une question au RAG via CLI

```bash
# Recherche vectorielle + réponse générée par Mistral
uv run python -m rag.chain "Je cherche un concert de jazz, tu as des idées ?"
```

## Mise en service de l'API

Prérequis : un jeton `AUTH_TOKEN` dans l'environnement, qui protège les routes consommant du quota Mistral.

```bash
# Mise en service
uv run uvicorn api.main:app --reload
```

Documentation interactive sur http://127.0.0.1:8000/docs, avec un bouton **Authorize** pour saisir le jeton.

| Méthode | Route | Rôle | Jeton |
|---|---|---|---|
| GET | `/health` | État de l'API et de l'index | non |
| GET | `/metadata` | Zone, période, volumes et modèles utilisés | non |
| POST | `/ask` | Question → réponse et sources | oui |
| POST | `/rebuild` | Reconstruction complète de l'index | oui |

### Poser une question

> Penser à mettre le jeton d'authentification à disposition dans l'environnement du terminal

```bash
# Bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -H "X-Token: $AUTH_TOKEN" \
  -d '{"question": "Quels concerts sont prévus à Toulouse ?"}'
```

```powershell
# PowerShell : curl.exe et guillemets échappés
curl.exe -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" -H "X-Token: $AUTH_TOKEN" `
  -d '{\"question\": \"Quels concerts sont prévus à Toulouse ?\"}'
```

Le champ optionnel `today` (`"2026-09-15"`) fixe la date de référence, pour rejouer une démonstration à date constante.

### Reconstruire l'index

```bash
# Bash
curl -X POST http://127.0.0.1:8000/rebuild -H "X-Token: $AUTH_TOKEN"
```

```powershell
# PowerShell
curl.exe -X POST http://127.0.0.1:8000/rebuild -H "X-Token: $AUTH_TOKEN"
```

Rejoue toute la chaîne (collecte, nettoyage, vectorisation) et recharge l'index sans redémarrer l'API. Compter 1 à 3 minutes et une consommation de quota Mistral.

## Docker

L'image intègre l'index `chunk_1000` donc le conteneur répond dès le démarrage, sans reconstruction.

```bash
# Construction de l'image
docker build -t puls-events .
# Mise en service
docker run --rm -p 8000:8000 --env-file .env puls-events
```

La clé Mistral et le jeton sont fournis **au lancement** via `--env-file`, jamais inscrits dans l'image. L'API est alors disponible sur http://127.0.0.1:8000/docs.

## Intégration et déploiement continue

`.github/workflows/ci.yml` vérifie chaque push : `ruff`, `pytest` avec un seuil de couverture à 80 %, et construction de l'image Docker. Les tests ne demandent ni clé API ni données brutes : ceux qui en dépendent s'ignorent automatiquement.

En cas de push sur `main`, un quatrième job déploie sur Render (uniquement si les trois vérifications sont OK).

> L'API est déployée à l'adresse : https://puls-events-api.onrender.com/docs

- Si l'instance est en sommeil elle redémarre en ~1 minutes
- La route `/rebuild` dépasse la limite de mémoire allouée à l'instance, elle n'est pas fonctionnelle sur ce endpoint.


## Évaluation

Le jeu de test annoté (20 questions, date de référence fixée) est dans `eval/test_set.json`.

```bash
# Pose les questions au RAG, calcule le hit@5 et sauvegarde les réponses > eval/results/<index>.json
uv run python -m rag.evaluate run no_chunk

# Evalue les scores Ragas (juge ministral-8b) à partir des réponses sauvegardées
uv run python -m rag.evaluate ragas no_chunk
```

Ragas 0.4.3 ne s'importe pas tel quel avec `langchain-community` 0.4 ([issue #2745](https://github.com/vibrantlabsai/ragas/issues/2745)). `rag/evaluate.py` applique un contournement avant l'import.

## Tests

```bash
# Lancement des tests
uv run pytest
# Avec génération du rapport
uv run pytest --cov=rag --cov-report=html
```
