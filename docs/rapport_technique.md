# Rapport technique - Assistant intelligent de recommandation d'événements culturels

## 1. Objectifs du projet

### Contexte
Puls-Events développe une plateforme de recommandations culturelles personnalisées. L'entreprise souhaite tester un **chatbot** capable de répondre aux questions des utilisateurs sur les événements culturels, à partir des données publiques **Open Agenda**.

### Problématique
Un LLM seul ne connaît pas les événements à venir d'une ville : ses connaissances s'arrêtent à sa date d'entraînement et il risque d'inventer des réponses. Un système **RAG (Retrieval-Augmented Generation)** répond à ce besoin :
1. il **recherche** les événements pertinents dans une base vectorielle construite à partir des données Open Agenda ;
2. il **fournit** ces événements au LLM comme contexte ;
3. le LLM **génère** une réponse en langage naturel, fondée sur des données réelles et à jour.

### Objectif du POC
Démontrer aux équipes produit et marketing :
- la **faisabilité technique** : une chaîne complète LangChain + Faiss + Mistral, exposée via une API REST et conteneurisée
- la **pertinence métier** : des réponses utiles et exactes sur des événements réels
- la **performance** : une qualité de réponse mesurée sur un jeu de test annoté

### Périmètre utilisé pour le POC
| Élément | Choix |
|---|---|
| Zone géographique | **Toulouse** |
| Période | Événements se terminant à partir du **01/09/2025** (un peu plus d'un an d'historique + événements à venir) |
| Données | Dataset public Open Agenda sur OpenDataSoft, agendas sans rapport avec la culture exclus |
| Volume | **3 868 événements** après nettoyage |
| Hors périmètre | Historique de conversation (une question → une réponse) |

---

## 2. Architecture du système

*À compléter (étapes 3 à 6).*

---

## 3. Préparation et vectorisation des données

Le cheminement complet (requêtes, graphiques, anomalies) est retracé dans le notebook [`scripts/eda_openagenda.ipynb`](../scripts/eda_openagenda.ipynb).

### 3.1 Source de données
- **Dataset** : [`evenements-publics-openagenda`](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda/), publié sur la plateforme OpenDataSoft (≈ 1,25 million d'événements, 56 champs).
- **API** : [Explore API v2.1](https://help.opendatasoft.com/apis/ods-explore-v2/explore_v2.1.html), sans clé. Les filtres s'écrivent en langage ODSQL.
- **Endpoint de collecte** : `/exports/json` qui renvoie tous les résultats en un appel.

### 3.2 Choix des filtres

| Étape | Filtre | Événements |
|---|---|---|
| Dataset complet | — | 1 254 053 |
| Événements récents | `lastdate_end >= date'2025-09-01'` | 339 145 |
| Zone géographique | `location_city = 'Toulouse'` | 5 241 |
| Agendas hors culture exclus | `originagenda_uid != …` (4 agendas) | **3 983** |

**Choix de la zone.** Une région représente plusieurs dizaines de milliers d'événements, ce qui est trop pour un POC (temps et coût de vectorisation). Toulouse présente une offre riche et dense avec un dataset plus contenu.

**Agendas exclus.** L'analyse des agendas sources de Toulouse montre une majorité d'agendas culturels (bibliothèques, musées, conservatoire, théâtres, patrimoine…). Quatre sont clairement hors sujet et sont exclus par leur identifiant (`originagenda_uid`) :

| Agenda | Événements | Motif |
|---|---|---|
| Mes événements France Travail | 1 125 | Emploi |
| Challenges Geovelo | 48 | Défis vélo |
| Semaine de l'industrie 2025 | 44 | Industrie |
| TM - Renov'energie | 37 | Rénovation énergétique |

### 3.3 Nettoyage

**Anomalies constatées et traitements** ([`rag/preprocess.py`](../rag/preprocess.py)) :

| Constat | Traitement | Lignes impactées |
|---|---|---|
| Titre vide | Événement supprimé | -13 |
| Evénements test (date > 2028) | Événement supprimé si début après le 31/12/2027 | -3 |
| Statut « Annulé » | Événement supprimé | -31 |
| Doublon (même titre, date de début et lieu) | Doublon supprimé | -69 |
| HTML dans la description longue | Nettoyage des balises & entités décodées | - |
| Mots-clés répétés | Dédoublonnés en conservant l'ordre | - |
| `status` et `attendancemode` en JSON multilingue | Extraction du libellé FR | - |
| Valeurs manquantes | Remplacées par une chaîne vide `""` | - |

Résultat : **3 868 événements conservés sur 3 983**.

**16 Champs conservés** (sur 56) : identifiant, titre, descriptions courte et longue, conditions, mots-clés, dates lisibles (`daterange_fr`), dates de début et de fin, nom, adresse et quartier du lieu, mode de participation, statut, agenda source, URL.

### 3.4 Construction des documents

Chaque événement nettoyé devient un `Document` LangChain ([`rag/documents.py`](../rag/documents.py)) composé de :

- **un texte à vectoriser**, structuré en lignes avec une architecture fixe :
  ```
  Titre : Visite guidée "Les peintres caravagesques en LSF"
  Dates : Dimanche 21 juin, 14h30 (2026)
  Lieu : Musée des Augustins, 21 rue de Metz 31000 Toulouse, Capitole / Arnaud Bernard / Carmes
  Description : Visite guidée thématique des collections en français. …
  ```
- **des métadonnées**, non vectorisées mais renvoyées avec chaque résultat : identifiant, titre, dates, lieu, statut, agenda, URL. Elles permettront au chatbot de citer ses sources.

Règles de construction :
- **Lieu et dates inclus dans le texte** : une question abordant une date ou un quartier doit pouvoir se rapprocher du document.
- **Année ajoutée aux dates qui l'omettent** : Open Agenda n'indique généralement pas l'année en cours dans le `datarange_fr`. On l'injecte dans le champ pour lever l'ambiguité.
- **Description courte non répétée** lorsqu'elle figure déjà au début de la description longue.
- **Format précisé uniquement s'il n'est pas « Sur place »** (« En ligne », « Mixte »).
- **Lignes vides omises** (mots-clés, conditions absents).

Longueur des textes obtenus : médiane de 737 caractères, 90 % sous 1 750 caractères, 280 documents au-delà de 2 000 caractères (maximum 9 327).

### 3.5 Chunking et embeddings

Concernant le **chunking**, son utilisation n'est ici pas une évidence. Un événement est déjà un bloc autonome (titre, dates, lieu, description) et le plus long document (≈ 2 700 tokens) tient dans le contexte de `mistral-embed` (8 000 tokens), le découpage n'est donc pas une contrainte technique. 

En revanche on peut argumenter que conserver des chunks trop longs peut avoir tendance à diluer le contenu et complexifier la recherche. Concernant le **choix du seuil** de découpe, il n'existe pas de consensus. Les valeurs courantes vont de 256 à 1 024 tokens avec 10 à 20 % de recouvrement. Le seuil de 1 000 caractères est une valeur qui permet de découper environ 30% de notre base, ce qui devrait permettre d'évaluer son intérêt.

Deux configurations sont donc construites et seront comparées lors de l'évaluation (§7) :
- **`no_chunk`** : un événement = un vecteur (référence) ;
- **`chunk_1000`** : chunks de 1 000 caractères, recouvrement de 150 (15 %).

**En-tête répété.** Un chunk isolé du milieu d'une description perdrait son association au titre, date et lieu. `split_documents` répète donc l'en-tête au début de chaque chunk. Les métadonnées sont également propagées dans chaque chunk.

**Fonctionnement du découpage.** `RecursiveCharacterTextSplitter` coupe le texte sur le premier séparateur présent (`\n\n`, puis `\n`, puis espace) et assemble les morceaux tant qu'ils tiennent dans `chunk_size`, et ne redescend au séparateur plus fin que si un morceau dépasse seul la limite.

**Embeddings.** Modèle `mistral-embed` : vecteurs de dimension 1 024, normalisés (norme 1), contexte de 8 000 tokens, 0,10 $ par million de tokens. Le même modèle doit vectoriser les documents et les questions : il est défini une seule fois (`EMBEDDING_MODEL` dans [`rag/index.py`](../rag/index.py)).

### 3.6 Limites connues des données
- Quelques petits agendas hors culture peuvent subsister mais restent minoritaires, les 30 plus gros agendas ayant été vérifiés.
- Les données sont statiques : les volumes ci-dessus correspondent à une collecte réalisée en septembre 2026.

---

## 4. Choix du modèle NLP

*À compléter (étape 4-A).*

---

## 5. Construction de la base vectorielle

**Construction des index** ([`rag/index.py`](../rag/index.py), `uv run python -m rag.index`) : deux configurations sont construites pour mesurer l'impact du découpage.

| Index | Découpage | Vecteurs | Durée | Taille (`index.faiss`) |
|---|---|---|---|---|
| `no_chunk` | Aucun (un événement = un vecteur) | 3 868 | 70 s | 15,8 Mo |
| `chunk_1000` | 1 000 caractères, recouvrement 150, en-tête répété | 6 216 | 125 s | 25,5 Mo |

- **Vectorisation** : `FAISS.from_documents` délègue à `MistralAIEmbeddings`, qui regroupe les textes par lots d'au plus 16 000 tokens et les envoie séquentiellement, avec relance automatique en cas d'erreur 429.
- **Coût** : 188 requêtes API au total (tests compris) pour 0,29$, prélevés sur le forfait mensuel de 10$ inclus dans l'offre gratuite Mistral. Les limites de débit de l'offre gratuite (1 requête/s, 20 M tokens/min pour `mistral-embed`) n'ont pas été atteintes.

**Algorithme d'indexation :**

Faiss propose plusieurs familles d'algorithmes, chacune faisant un compromis entre vitesse, mémoire et exactitude :

| Algorithme | Principe | Usage type |
|---|---|---|
| **Flat** | Compare la question à tous les vecteurs (recherche exacte) | Jusqu'à quelques dizaines de milliers de vecteurs |
| **IVF** | Regroupe les vecteurs par k-means et recherche dans les groupes les plus proches (`nprobe`) | Centaines de milliers à millions de vecteurs |
| **HNSW** | Parcourt un graphe de voisins multi-niveaux | Millions de vecteurs, recherche très rapide et précise |
| **PQ** | Quantification des vecteurs | Réduction de la mémoire, se combine à IVF |

Un benchmark a été réalisé pour évaluer les performances des différentes approches sur les 6 216 vecteurs de `chunk_1000` ([`scripts/benchmark_faiss.py`](../scripts/benchmark_faiss.py). Ce benchmark comporte **500 questions simulées** (bruit ajouté aux vecteurs existants de la base) qui ne nécessitent pas d'appel API. Le **rappel@5** est la part des 5 vrais plus proches voisins retrouvés :

| Index | Construction | Recherche / question | Rappel@5 | Taille |
|---|---|---|---|---|
| **Flat** (retenu) | 6 ms | 0,043 ms | **100 %** | 25,5 Mo |
| HNSW32 | 258 ms | 0,030 ms | 100 % | 27,1 Mo |
| IVF80, nprobe=8 | 119 ms | 0,045 ms | 98 % | 25,8 Mo |
| IVF80,PQ64 | **41 s** | 0,035 ms | 80 % | 1,8 Mo |

A cette échelle, on observe une variation d'un centième de milliseconde, ce temps est négligeable face aux appels réseau vers Mistral. `IndexFlatL2` est donc le choix optimal : exact et sans paramètre à régler. HNSW ou IVF deviendraient potentiellement pertinents pour un passage à l'échelle (340 k événements récents en France).

**Métrique :** Faiss renvoie le carré de la distance euclidienne (plus petit = plus proche). Les vecteurs Mistral étant normalisés, ce classement est identique à celui de la similarité cosinus (cos = 1 − d² / 2).

**Persistance :** Chaque index est sauvegardé dans `data/index/<configuration>/` (non versionné) :
- `index.faiss` : les vecteurs bruts (1 024 flottants de 4 octets par vecteur, plus un en-tête)
- `index.pkl` : les textes et métadonnées associés à chaque vecteur, au format pickle

**Tests** ([`tests/test_index.py`](../tests/test_index.py)) : pour chaque index, nombre de vecteurs égal au nombre de chunks, dimension 1 024, et recherche d'un titre connu renvoyant l'événement en premier avec ses métadonnées.

---

## 6. API et endpoints exposés

*À compléter (étape 5).*

---

## 7. Évaluation du système

*À compléter (étape 4-B).*

---

## 8. Recommandations et perspectives

*À compléter (étape 6-B).*

---

## 9. Organisation du dépôt GitHub

```
P7/
├── rag/                      # Logique métier, réutilisée par les scripts, l'API et les tests
│   ├── collect.py            # Collecte des événements Open Agenda -> data/raw/
│   ├── preprocess.py         # Nettoyage des événements -> data/processed/
│   ├── documents.py          # Construction des Documents LangChain et découpage en chunks
│   └── index.py              # Vectorisation Mistral et index Faiss -> data/index/
├── scripts/
│   ├── check_env.py          # Vérification des imports et de la clé API Mistral
│   ├── benchmark_faiss.py    # Comparaison des algorithmes d'index Faiss (Flat, HNSW, IVF, PQ)
│   └── eda_openagenda.ipynb  # Analyse exploratoire justifiant la collecte et le nettoyage
├── tests/                    # Tests unitaires (pytest)
├── docs/
│   └── rapport_technique.md  # Ce rapport
├── data/                     # Données générées, non versionnées (raw/, processed/, index/)
├── pyproject.toml / uv.lock  # Dépendances (gestionnaire uv)
└── README.md                 # Installation et commandes
```

---

## 10. Annexes

*À compléter.*
