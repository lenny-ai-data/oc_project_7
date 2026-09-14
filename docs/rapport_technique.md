# Rapport technique - Assistant intelligent de recommandation d'événements culturels

> POC réalisé pour **Puls-Events**.

---

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

Longueur des textes obtenus : médiane de 732 caractères, 90 % sous 1 750 caractères, 280 documents au-delà de 2 000 caractères (maximum 9 320).

### 3.5 Chunking et embeddings

*À compléter (étape 3).*

### 3.6 Limites connues des données
- Quelques petits agendas hors culture peuvent subsister mais restent minoritaires, les 30 plus gros agendas ayant été vérifiés.
- Les données sont statiques : les volumes ci-dessus correspondent à une collecte réalisée en septembre 2026.

---

## 4. Choix du modèle NLP

*À compléter (étape 4-A).*

---

## 5. Construction de la base vectorielle

*À compléter (étape 3).*

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
│   └── documents.py          # Construction des Documents LangChain (texte + métadonnées)
├── scripts/
│   ├── check_env.py          # Vérification des imports de l'environnement
│   └── eda_openagenda.ipynb  # Analyse exploratoire justifiant la collecte et le nettoyage
├── tests/                    # Tests unitaires (pytest)
├── docs/
│   └── rapport_technique.md  # Ce rapport
├── data/                     # Données générées, non versionnées (raw/, processed/)
├── pyproject.toml / uv.lock  # Dépendances (gestionnaire uv)
└── README.md                 # Installation et commandes
```

---

## 10. Annexes

*À compléter.*
