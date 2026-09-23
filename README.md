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

## API

Prérequis : un jeton dans le `.env` (`AUTH_TOKEN=...`), qui protège les routes consommant du quota Mistral.

```bash
uv run uvicorn api.main:app --reload
```

Documentation interactive sur http://127.0.0.1:8000/docs, avec un bouton **Authorize** pour saisir le jeton.

| Méthode | Route | Rôle | Jeton |
|---|---|---|---|
| GET | `/health` | État de l'API et de l'index | non |
| GET | `/metadata` | Zone, période, volumes et modèles utilisés | non |
| POST | `/ask` | Question en langage naturel → réponse et sources | oui |
| POST | `/rebuild` | Reconstruction complète de l'index | oui |

### Poser une question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -H "X-Token: $AUTH_TOKEN" \
  -d '{"question": "Quels concerts sont prévus à Toulouse ?"}'
```

> Sous PowerShell, `curl` est un alias d'`Invoke-WebRequest` : utiliser `curl.exe`.

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "Quels concerts sont prévus à Toulouse ?"},
    headers={"X-Token": "votre-jeton"},
)
result = response.json()

print(result["answer"])
for source in result["sources"]:
    print(f"- {source['title']} ({source['date_range']}, {source['location_name']})")
```

> Les données contiennent des caractères typographiques français absents de la page de code Windows : sur une console `cp1252`, lancer le script avec `PYTHONIOENCODING=utf-8` pour éviter une `UnicodeEncodeError`.

Le champ optionnel `today` (`"2026-09-15"`) fixe la date de référence, pour rejouer une démonstration à date constante.

### Reconstruire l'index

```bash
curl -X POST http://127.0.0.1:8000/rebuild -H "X-Token: $AUTH_TOKEN"
```

Rejoue toute la chaîne (collecte, nettoyage, vectorisation) et recharge l'index sans redémarrer l'API. Compter quelques minutes et une consommation de quota Mistral.

> Cette route demande environ 200 Mo de mémoire au pic : elle fonctionne en local et dans le conteneur, mais dépasse la limite d'une petite instance en ligne (voir §6.7 du rapport).

## Docker

L'image embarque l'index `chunk_1000`, donc le conteneur répond dès le démarrage, sans reconstruction.

```bash
docker build -t puls-events .
docker run --rm -p 8000:8000 --env-file .env puls-events
```

La clé Mistral et le jeton sont fournis **au lancement** via `--env-file`, jamais inscrits dans l'image. L'API est alors disponible sur http://127.0.0.1:8000/docs.

## Intégration continue

`.github/workflows/ci.yml` vérifie chaque push : `ruff`, `pytest` avec un seuil de couverture à 80 %, et construction de l'image Docker. Un quatrième job déploie sur Render, uniquement depuis `main` et uniquement si les trois vérifications sont vertes.

Les tests ne demandent ni clé API ni données brutes : ceux qui en dépendent s'ignorent automatiquement.

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
# Avec rapport
uv run pytest --cov=rag --cov-report=html
```
