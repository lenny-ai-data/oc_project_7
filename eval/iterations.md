# Itérations d'évaluation

Jeu de test : `eval/test_set.json` (20 questions, date de référence 15/09/2026).
Hit@5 sur les 15 questions qui attendent des événements ; scores Ragas sur ces mêmes 15 questions.
Le juge est passé de `ministral-8b` à `ministral-14b` à l'itération 4 (voir §4) : **les scores Ragas ne sont comparables qu'à juge identique**. 

## Tableau de bord

| # | Changement | Index | Juge | Hit@5 | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|---|---|
| 1 | Référence | `no_chunk` | 8b | 87 % | 0,67 | 0,62 | 0,48 | 0,58 |
| 1 | Référence | `chunk_1000` | 8b | **93 %** | 0,76 | 0,56 | 0,46 | 0,57 |
| 2 | Prompt : week-end seulement si demandé | `no_chunk` | 8b | 87 % | 0,84 | 0,75 | 0,51 | 0,67 |
| 2 | Prompt : week-end seulement si demandé | `chunk_1000` | 8b | **93 %** | 0,78 | 0,74 | 0,45 | 0,57 |
| 3 | Découpage : conditions dans chaque chunk | `chunk_1000` | 8b | 93 % | 0,86 | 0,80 | 0,53 | 0,65 |
| 3 | *(mêmes résultats, renotés pour comparaison)* | `chunk_1000` | 14b | 93 % | 0,865 | 0,812 | 0,479 | 0,643 |
| 4 | Filtrage des dates avant la recherche | `chunk_1000` | 14b | **100 %** | 0,86 | 0,81 | 0,50 | 0,670 |
| 5 | Bornes de la période extraite | `chunk_1000` | 14b | **100 %** | **0,96** | 0,76 | 0,47 | **0,71** |

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
- **Effet de bord fortement réduit** : les réponses qui parlent du week-end sans que la question le demande passent de 8 à 3 (`no_chunk`) et de 10 à 5 (`chunk_1000`).
- **Réponses** : *faithfulness* +0,17 et *answer relevancy* +0,13 pour `no_chunk`, *answer relevancy* +0,18 pour `chunk_1000`. Les concerts de Noël, les films espagnols et la sortie avec un bébé sont maintenant proposés.
- **Recherche inchangée** : le hit@5 reste identique (87 % et 93 %, mêmes échecs), ce qui est attendu puisque seul le prompt a changé.
- **Bruit du juge** : *context precision* et *context recall* ne dépendent que des contextes et de la référence, qui n'ont pas changé. Leurs moyennes bougent pourtant (*context recall* de `no_chunk` : 0,58 → 0,67).
- **Limites restantes** :
  - `temp-01` et `temp-02` (`no_chunk`) échouent dès la recherche : le système répond honnêtement « rien ce week-end », sans aider l'utilisateur. Côté `chunk_1000`, `temp-01` présente une visite de campus comme un « atelier-concert ».
  - `info-01` : un « je ne connais pas le tarif » correct obtient une *answer relevancy* de 0, car Ragas note 0 toute réponse jugée évasive.
  - `fact-04` (`chunk_1000`) : les conditions sont toujours absentes du chunk retenu (itération 3).
  - `hors-01` et `hors-02` : refus, suivis d'un recadrage vers les événements du week-end, cohérent pour un assistant de recommandation.

## 3. Découpage : conditions d'accès répétées dans chaque chunk

**Changement** (`rag/documents.py`) : la ligne `Conditions` (tarif, réservation), en fin de texte, est déplacée dans l'en-tête répété de chaque chunk. L'index `chunk_1000` est reconstruit (6 273 chunks au lieu de 6 216).

**Constats**
- **Cause corrigée** : aucun chunk n'est plus privé des conditions de son événement (1 687 auparavant).
- **`fact-04` résolu** : « Non, il n'est pas nécessaire de réserver… gratuite et sans réservation, limitée à 80 personnes ».
- **Moyennes** : *faithfulness* 0,78 → 0,86, *answer relevancy* 0,74 → 0,80, *context precision* 0,45 → 0,53, *context recall* 0,57 → 0,65. L'amélioration est générale mais chaque écart reste dans le bruit du juge.
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

- **Hit@5 : 93 % → 100 %**, catégorie temporelle **67 % → 100 %**. Sur `temp-01`, le top-5 passe de 0 à 2 événements attendus.
- **Aucune dégradation détectable de la génération** : à juge égal, *faithfulness* passe de 0,865 à 0,86 et *answer relevancy* de 0,812 à 0,81, soit moins que leur bruit propre (respectivement 0,055 et 0,056 d'écart moyen entre deux exécutions identiques).
- **Juge remplacé, avec mesure à l'appui.** Sur des contextes rigoureusement identiques, `ministral-8b` variait jusqu'à 1,00 par question (écart moyen 0,081 en *context precision*, 0,103 en *context recall*). Sur trois exécutions, `ministral-14b` rend un *context recall* **identique 15 fois sur 15**. Coût : 4,5 min au lieu de 2, avec `MAX_WORKERS = 2` pour la limite de 30 req/min.
- **Seul le *context recall* est exploitable pour comparer deux itérations.** Il monte de 0,643 à 0,670, et **uniquement sur `temp-01` (0 → 0,25) et `temp-02` (0,40 → 0,60)**, exactement les questions dont la récupération a changé.

**Limites restantes**

- ***Context precision* reste peu fiable.** Elle varie de 0,50 sur `temp-03` d'une exécution à l'autre, et surtout elle **dépend de la réponse** : sur `reco-02`, question, contextes et référence identiques, seule la réponse diffère, et le score passe de 0,70 à 0,20.
- **L'annotation par exemples plafonne les métriques de contexte.** `temp-01` accepte 7 identifiants mais n'en nomme que trois dans sa référence, les deux événements retrouvés sont valides sans y figurer, d'où un *context recall* qui reste bas malgré une réponse juste.
- **`info-01`** conserve une *answer relevancy* de 0 : Ragas pénalise toute réponse jugée évasive, y compris un « je ne dispose pas du tarif » exact.

## 5. Bornes de la période extraite

**Problème** : en préparant la démonstration, « Quels concerts de Noël sont prévus en décembre ? » répondait qu'aucun n'était prévu, alors que l'index en contient trois (13, 17 et 18 décembre). Le jeu de test ne le détectait pas : aucune de ses questions ne vise un mois entier.

**Diagnostic** : sortie brute de l'extraction, à la date du 24/09/2026.

- **Fin de mois omise** : « en novembre » et « en décembre » rendent `debut` = 1er du mois et `fin = None`. Le repli réduit alors la période au premier jour et la recherche ne porte que sur le 1er du mois. Seul « en octobre », cité en exemple dans la consigne, recevait ses deux bornes.
- **Année passée** : « une sortie en février ? » rendait février 2026, déjà écoulé.

**Changement** (`rag/chain.py`) : deux lignes de consigne et la description du champ `fin`, sans logique nouvelle.

- « Donne toujours les deux bornes : un mois couvre tous ses jours, « en octobre » va du 1er au 31 octobre. »
- « Sans année précisée, la période est la prochaine à venir : jamais une période déjà passée. »
- `fin` : « Dernier jour inclus, toujours renseigné si situe_dans_le_temps est vrai (égal à debut pour un seul jour) ».

**Constats**

- **Sur 11 questions de contrôle**, toutes les périodes sont justes : mois entiers, février 2027, week-end, lendemain, Toussaint, et aucune période pour « un concert de jazz ».
- **Hit@5 inchangé à 100 %.** Seules trois questions changent de sources (`temp-03`, `none-01`, `none-02`), toutes porteuses d'une indication temporelle.
- **Ragas dans le bruit du juge** : *context recall* 0,670 → 0,71, mais le seul écart est `fact-02` (0,5 → 1,0), dont les sources n'ont pas changé. *Answer relevancy* 0,81 → 0,76, porté par `reco-01` (0,78 → 0), dont la réponse est **identique mot pour mot**. *Faithfulness* 0,86 → 0,96, porté par `reco-01` et `info-01`, aux réponses également inchangées.
- **Effets de la consigne sur le jeu de test** :

| Question | Avant | Après | |
|---|---|---|---|
| `reco-03` concerts de Noël gratuits | aucune | 1er → 31 déc. | cohérent |
| `none-01` « dans les prochains mois » | aucune | 16 sept. → 31 déc. | cohérent |
| `temp-03` « fin octobre » | aucune | 1er → 31 oct. | trop large, mieux qu'aucun filtre |
