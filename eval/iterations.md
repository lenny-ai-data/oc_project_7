# Itérations d'évaluation

Jeu de test : `eval/test_set.json` (20 questions, date de référence 15/09/2026).
Hit@5 sur les 15 questions qui attendent des événements ; scores Ragas sur ces mêmes 15 questions.
Le juge est passé de `ministral-8b` à `ministral-14b` à l'itération 4 (voir §4) : **les scores Ragas ne sont comparables qu'à juge identique**, d'où la ligne « 3 renotée ».
Les résultats détaillés de chaque itération sont dans l'historique git de `eval/results/`.

## Tableau de bord

| # | Changement | Index | Juge | Hit@5 | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|---|---|
| 1 | Référence | `no_chunk` | 8b | 87 % | 0,67 | 0,62 | 0,48 | 0,58 |
| 1 | Référence | `chunk_1000` | 8b | 93 % | 0,76 | 0,56 | 0,46 | 0,57 |
| 2 | Prompt : week-end seulement si demandé | `no_chunk` | 8b | 87 % | **0,84** | **0,75** | 0,51 | 0,67 |
| 2 | Prompt : week-end seulement si demandé | `chunk_1000` | 8b | 93 % | 0,78 | **0,74** | 0,45 | 0,57 |
| 3 | Découpage : conditions dans chaque chunk | `chunk_1000` | 8b | 93 % | **0,86** | **0,80** | 0,53 | 0,65 |
| 3 | *(mêmes résultats, renotés pour comparaison)* | `chunk_1000` | 14b | 93 % | 0,865 | 0,812 | 0,479 | 0,643 |
| 4 | Filtrage des dates avant la recherche | `chunk_1000` | 14b | **100 %** | 0,86 | 0,81 | 0,50 | **0,670** |

## 1. Référence

**Configuration** : `ministral-14b`, top 5, filtre des événements terminés, date du jour et week-end dans le prompt.

**Constats**
- **Questions « ce week-end »** : `temp-01` échoue sur les deux index. La similarité ignore les dates et ramène des concerts de novembre à avril.
- **Effet de bord du prompt** : 8/20 (`no_chunk`) et 10/20 (`chunk_1000`) réponses restreignent au week-end des questions qui n'en parlent pas (« aucun concert de Noël ce week-end »). Le juge les classe comme évasives, d'où des *answer relevancy* à 0.
- **Découpage** : 1 687 chunks sur 6 216 ne contiennent pas la ligne `Conditions` de leur événement. Exemple : `fact-04` répond « je ne sais pas » au lieu de « sans réservation ».
- **Juge** : 2 scores manquants (JSON mal formé par `ministral-8b`), exclus des moyennes.

## 2. Prompt : dates du week-end utilisées seulement si la question en parle

**Changement** (`rag/chain.py`) : « ce week-end désigne le … » devient « Si la question parle du week-end, il s'agit du … ; sinon, ne limite pas ta réponse à une période. »

**Constats**
- **Effet de bord fortement réduit** : les réponses qui parlent du week-end sans que la question le demande passent de 8 à 3 (`no_chunk`) et de 10 à 5 (`chunk_1000`). Ce comptage est automatique et majore le problème : `fact-05`, par exemple, cite à juste titre le 19-20 septembre, qui est la date de l'exposition.
- **Réponses** : *faithfulness* +0,17 et *answer relevancy* +0,13 pour `no_chunk`, *answer relevancy* +0,18 pour `chunk_1000`. Les concerts de Noël, les films espagnols et la sortie avec un bébé sont maintenant proposés.
- **Recherche inchangée** : le hit@5 reste identique (87 % et 93 %, mêmes échecs), ce qui est attendu puisque seul le prompt a changé.
- **Bruit du juge** : *context precision* et *context recall* ne dépendent que des contextes et de la référence, qui n'ont pas changé. Leurs moyennes bougent pourtant (*context recall* de `no_chunk` : 0,58 → 0,67 ; `info-01` passe de 0 à 1). Avec `ministral-8b`, **un écart de moins de 0,1 n'est pas significatif** : seules les tendances nettes comptent.
- **Limites restantes** :
  - `temp-01` et `temp-02` (`no_chunk`) échouent dès la recherche : le système répond honnêtement « rien ce week-end », sans aider l'utilisateur. Côté `chunk_1000`, `temp-01` présente une visite de campus comme un « atelier-concert ».
  - `info-01` : un « je ne connais pas le tarif » correct obtient une *answer relevancy* de 0, car Ragas note 0 toute réponse jugée évasive.
  - `fact-04` (`chunk_1000`) : les conditions sont toujours absentes du chunk retenu (itération 3).
  - `hors-01` et `hors-02` : refus, suivis d'un recadrage vers les événements du week-end, cohérent pour un assistant de recommandation.

## 3. Découpage : conditions d'accès répétées dans chaque chunk

**Changement** (`rag/documents.py`) : la ligne `Conditions` (tarif, réservation), en fin de texte, est déplacée dans l'en-tête répété de chaque chunk. L'index `chunk_1000` est reconstruit (6 273 chunks au lieu de 6 216) ; `no_chunk` n'est pas concerné.

**Constats**
- **Cause corrigée** : aucun chunk n'est plus privé des conditions de son événement (1 687 auparavant).
- **`fact-04` résolu** : « Non, il n'est pas nécessaire de réserver… gratuite et sans réservation, limitée à 80 personnes ». Ses scores passent de 0,40 / 0 / 0 / 0,50 à 0,83 / 0,86 / 1,0 / 1,0.
- **Moyennes** : *faithfulness* 0,78 → 0,86, *answer relevancy* 0,74 → 0,80, *context precision* 0,45 → 0,53, *context recall* 0,57 → 0,65. L'amélioration est générale ; hors `fact-04`, chaque écart reste dans le bruit du juge. Exemple de bruit : `fact-01` a les mêmes sources, mais son *context recall* passe de 0,5 à 1,0.
- **Recherche** : le hit@5 reste à 93 %, mais les sources changent pour 10 questions sur 15. Les en-têtes plus riches modifient les vecteurs des chunks.
- **Limites restantes** :
  - `temp-01` (« concerts gratuits ce week-end ») échoue toujours : la recherche ignore les dates.
  - *Context precision* et *context recall* restent à 0 sur les questions larges (`reco-01`, `temp-01` à `temp-03`) : la référence cite quelques exemples, et les événements retrouvés, parfois tout aussi valables, n'y figurent pas. C'est une limite de l'annotation par exemples pour les questions ouvertes.
  - `info-01` : réponse correcte (« je ne dispose pas du tarif »), mais *answer relevancy* reste à 0 (réponse jugée évasive), et le modèle suggère encore de consulter le site du Château d'Eau.

## 4. Filtrage des dates avant la recherche vectorielle

**Changement** (`rag/chain.py`) : trois modifications indissociables.

- Les dates sont extraites du docstore au chargement, en **tableaux numpy alignés sur les positions Faiss**, et le filtre s'applique **avant** la recherche via `IDSelectorBatch`. Le wrapper LangChain, lui, ne sait filtrer qu'après.
- La **période visée par la question est extraite par le LLM** (`with_structured_output`), avec le week-end calculé en Python et des contre-exemples explicites dans la consigne.
- `FETCH_K = 200` devient `K_CHUNKS = 100` : tous les chunks rendus étant déjà valides, la sur-recherche disparaît. 100 garantit 5 événements distincts, un événement occupant au plus 18 chunks.

Le juge Ragas passe de `ministral-8b` à `ministral-14b` (voir plus bas).

**Constats**

- **Neutralité du pré-filtre vérifiée avant d'ajouter le self-query.** Comparé au post-filtrage sur les 20 questions, il rend **19 fois exactement les mêmes événements dans le même ordre**. La vingtième est un ex æquo — distances 0,467780 et 0,467880 — que le bruit de vectorisation fait basculer : `mistral-embed` ne renvoie pas le même vecteur d'un appel à l'autre (écarts jusqu'à 9·10⁻⁴). **Le hit@5 a donc lui aussi un plancher de bruit** quand un événement attendu est à la frontière du top-5.
- **Pourquoi les deux parties sont indissociables.** Le filtre imposé garde 15 % des chunks ; un filtre « ce week-end » n'en garde que 1,5 %. Sur 200 candidats, une période aussi serrée n'aurait laissé que 3 événements en espérance, sous les 5 demandés. Élargir `fetch_k` sans self-query n'aurait rien changé ; le self-query sans pré-filtre aurait été tronqué.
- **Hit@5 : 93 % → 100 %**, catégorie temporelle **67 % → 100 %**. Sur `temp-01`, le top-5 passe de 0 à 2 événements attendus.
- **Le gain est d'abord qualitatif.** « Des concerts gratuits à voir ce week-end ? » répondait « aucun concert gratuit ce week-end » alors qu'il en existait ; elle cite maintenant un concert réel du dimanche 20 septembre.
- **Génération inchangée** : *faithfulness* 0,865 → 0,86 et *answer relevancy* 0,812 → 0,81 à juge égal. C'était le risque principal — améliorer la recherche en dégradant les réponses.
- **Juge remplacé, avec mesure à l'appui.** Sur des contextes rigoureusement identiques, `ministral-8b` variait jusqu'à 1,00 par question (écart moyen 0,081 en *context precision*, 0,103 en *context recall*). Sur trois exécutions, `ministral-14b` rend un *context recall* **identique 15 fois sur 15**. Coût : 4,5 min au lieu de 2, avec `MAX_WORKERS = 2` pour la limite de 30 req/min.
- **Seul le *context recall* est exploitable pour comparer deux itérations.** Il monte de 0,643 à 0,670, et **uniquement sur `temp-01` (0 → 0,25) et `temp-02` (0,40 → 0,60)** — exactement les questions dont la récupération a changé.

**Limites restantes**

- ***Context precision* reste inutilisable.** Elle varie de 0,50 sur `temp-03` d'une exécution à l'autre, et surtout elle **dépend de la réponse** malgré son nom : sur `reco-02`, question, contextes et référence identiques, seule la réponse diffère, et le score passe de 0,70 à 0,20.
- **L'annotation par exemples plafonne les métriques de contexte.** `temp-01` accepte 7 identifiants mais n'en nomme que trois dans sa référence ; les deux événements retrouvés sont valides sans y figurer, d'où un *context recall* qui reste bas malgré une réponse juste.
- **Un appel LLM de plus par question**, donc davantage de latence et de quota. L'extraction se désactive proprement si le modèle ne sait pas produire de sortie structurée.
- **`info-01`** conserve une *answer relevancy* de 0 : Ragas pénalise toute réponse jugée évasive, y compris un « je ne dispose pas du tarif » exact.

## Bilan provisoire

| Index | Itération | Hit@5 | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|
| `no_chunk` | 2 | 87 % | 0,84 | 0,75 | 0,51 | 0,67 |
| `chunk_1000` | 3 | 93 % | 0,86 | 0,80 | 0,53 | 0,65 |
| `chunk_1000` | **4** | **100 %** | 0,86 | 0,81 | 0,50 | **0,67** |

`chunk_1000` devance légèrement `no_chunk`. Le seul écart net est le hit@5 (une question de plus retrouvée, `temp-02`) ; les écarts Ragas restent dans le bruit du juge.
