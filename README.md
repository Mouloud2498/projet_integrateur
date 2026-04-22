# BooksMatch — Système Intelligent de Recommandation de Livres

> Projet de science des données et de développement logiciel — IFM30542-H2026 · Groupe 7 · Collège La Cité, Ottawa

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Dataset](#2-dataset)
3. [Structure du projet](#3-structure-du-projet)
4. [Pipeline du notebook](#4-pipeline-du-notebook)
5. [Modèles de recommandation](#5-modèles-de-recommandation)
6. [Logique hybride finale](#6-logique-hybride-finale)
7. [Évaluation](#7-évaluation)
8. [Interprétabilité avec SHAP](#8-interprétabilité-avec-shap)
9. [Application web](#9-application-web)
10. [Artefacts sauvegardés](#10-artefacts-sauvegardés)
11. [Installation et configuration](#11-installation-et-configuration)
12. [Utilisation](#12-utilisation)
13. [Limites identifiées](#13-limites-identifiées)
14. [Pistes d'amélioration](#14-pistes-damélioration)
15. [Conclusion](#15-conclusion)

---

## 1. Vue d'ensemble du projet

**BooksMatch** est un système de recommandation de livres construit de bout en bout, depuis l'exploration des données brutes jusqu'au déploiement d'une application web interactive. Le projet a été développé dans le cadre du cours IFM30542-H2026 au Collège La Cité, et répond à deux objectifs principaux.

Le premier objectif est de proposer des **recommandations personnalisées** à chaque utilisateur, en tenant compte de son historique de lecture et de ses préférences. Pour les utilisateurs avec suffisamment d'interactions, le système exploite un modèle de filtrage collaboratif (SVD) combiné à une représentation sémantique des livres (embeddings). Pour les utilisateurs avec peu ou pas d'historique, le système se replie sur des stratégies plus robustes basées sur le contenu et la popularité.

Le second objectif est de permettre la **découverte de livres similaires**. Quel que soit le statut de l'utilisateur, il est possible d'obtenir les livres les plus proches d'un titre donné, en s'appuyant sur la similarité cosinus entre les représentations vectorielles des livres.

BooksMatch est conçu pour fonctionner dans des conditions de données difficiles : faible couverture des métadonnées, matrice d'interactions très sparse, et profils utilisateurs souvent légers. Ces contraintes ont guidé l'ensemble des choix architecturaux du projet.

---

## 2. Dataset

**Source :** [Kaggle — Book Recommender System Dataset](https://www.kaggle.com/)

| Indicateur | Valeur |
|---|---|
| Livres dans le catalogue | 898 |
| Interactions totales | 196 296 |
| Utilisateurs uniques | 66 909 |
| Livres avec métadonnées exploitables | 96 (≈ 10,7 %) |
| Sparsité de la matrice | ≈ 99,67 % |

### Pourquoi ce dataset est-il difficile ?

En apparence, 196 296 interactions semblent confortables. En réalité, deux problèmes structurels se révèlent dès l'analyse exploratoire.

**La sparsité extrême** est le premier problème. Avec 66 909 utilisateurs et 898 livres, la matrice complète contiendrait environ 60 millions de cases. Seulement 196 296 sont remplies — soit 0,33 % du total. La majorité des utilisateurs n'ont évalué que quelques livres, rendant les profils individuels très peu informatifs pour un modèle collaboratif.

**La pauvreté des métadonnées** est le second problème. Sur les 898 livres, seulement 96 disposent d'une description complète, d'un genre défini et d'un auteur renseigné. Les 89,3 % restants ne sont connus que par leur titre, ce qui limite fortement la capacité d'un modèle content-based à discriminer les livres entre eux.

Ces deux contraintes expliquent pourquoi une approche hybride, capable de combiner plusieurs signaux et de se dégrader gracieusement, était nécessaire.

---

## 3. Structure du projet

```
booksmatch/
│
├── notebook/
│   └── Final_Final.ipynb          # Pipeline complet : données → artefacts
│
├── models/                        # Artefacts sauvegardés après entraînement
│   ├── svd_model.pkl
│   ├── book_embeddings_df.pkl
│   ├── embedding_matrix.npy
│   ├── books_base_df_final.pkl
│   ├── user_profiles_df_final.pkl
│   ├── train_df.pkl
│   ├── test_df.pkl
│   ├── popular_books_df.pkl
│   └── hybrid_config.json
│
├── templates/
│   └── index.html                 # Interface HTML principale
│
├── static/
│   ├── style.css                  # Styles CSS
│   ├── script.js                  # Logique frontend JavaScript
│   └── logo.jpg                   # Logo BooksMatch
│
├── app.py                         # Backend Flask
├── requirements.txt               # Dépendances Python
└── README.md
```

---

## 4. Pipeline du notebook

Le notebook constitue le cœur scientifique du projet. Il couvre l'intégralité du workflow, de l'ingestion des données brutes à la sauvegarde des artefacts prêts à être utilisés par l'application Flask. Chaque étape a été conçue non pas comme une opération isolée, mais comme une brique qui conditionne les décisions suivantes.

### 4.1 Chargement et inspection des données

La première étape consiste à charger les deux sources principales — les interactions utilisateur-livre et les métadonnées des livres — et à les inspecter soigneusement. Cette inspection révèle des colonnes parasites, des types incorrects et des doublons dans les métadonnées. Deux tables propres sont alors construites : `ratings_df` pour la partie collaborative et `books_df` pour la partie descriptive.

**Pourquoi c'est important :** Les erreurs introduites lors du nettoyage se propagent dans tout le pipeline. Prendre le temps de comprendre les données avant de les utiliser évite des diagnostics tardifs et coûteux.

### 4.2 Analyse exploratoire

L'analyse exploratoire confirme les deux constats fondamentaux du projet : la sparsité extrême de la matrice et la faible couverture des métadonnées. Elle révèle également que les notes sont majoritairement positives, avec une moyenne proche de 4/5, et que la médiane d'activité utilisateur est très faible — la majorité des profils ne contiennent que quelques interactions.

**Ce qu'on en apprend :** Un modèle collaboratif pur sera fragile sur la majorité des profils. La popularité sera une baseline difficile à battre.

### 4.3 Filtrage K-core

Le filtrage K-core consiste à retirer de la matrice les utilisateurs et les livres qui n'atteignent pas un seuil minimum d'interactions, de façon itérative. Cette opération élimine le bruit introduit par les profils ultra-légers, sans sacrifier la couverture du catalogue.

Après filtrage : **146 410 interactions**, **32 776 utilisateurs**, **898 livres** (catalogue intact).

**Pourquoi c'est important :** Un modèle SVD entraîné sur des données très bruitées apprend mal. Le filtrage améliore la qualité du signal collaboratif sans réduire les livres recommandables.

### 4.4 Construction du catalogue enrichi (books_base_df)

Plutôt que de travailler avec deux tables séparées, une table unifiée `books_base_df` est construite. Elle part de tous les livres des interactions (898) et y joint les métadonnées disponibles, les statistiques de notes, et un **score de popularité inspiré d'IMDb**.

Ce score pondère la note moyenne du livre vers la note globale du catalogue, proportionnellement au nombre d'évaluations. Un livre peu évalué ne peut pas obtenir un score élevé même s'il a une note parfaite, ce qui évite de propulser des livres confidentiels au sommet des recommandations.

### 4.5 Nettoyage textuel et qualité des métadonnées

Les textes bruts des métadonnées sont nettoyés en profondeur : suppression des balises HTML, normalisation des espaces, extraction des numéros de série depuis les titres, gestion des valeurs vides. Un texte final structuré (`book_text_v2`) est produit pour chaque livre.

Chaque livre est également classé selon la richesse de son contenu : `rich` (description + genre + auteur, 10,7 %) ou `poor` (titre seul ou données partielles, 89,3 %). Cette classification est essentielle pour interpréter les limites du volet content-based.

### 4.6 Génération des embeddings Jina v4

Les textes nettoyés sont encodés avec **Jina Embeddings v4**, un modèle de représentation sémantique moderne. Chaque livre est transformé en un vecteur dense de 2 048 dimensions. La proximité entre deux vecteurs dans cet espace correspond à la similarité sémantique entre les deux livres.

**Pourquoi Jina v4 :** Ce modèle est conçu pour capturer des relations sémantiques fines, y compris entre des textes courts comme des titres de livres. Il fonctionne sur l'ensemble du catalogue, même pour les livres sans description.

> **Note :** La génération des embeddings nécessite une clé API Jina (`JINA_API_KEY`). Voir la section Installation.

### 4.7 Similarité cosinus pour les recommandations livre-livre

Une matrice de similarité cosinus est calculée entre tous les embeddings. Cette matrice permet de retrouver, pour n'importe quel livre du catalogue, les K livres les plus proches dans l'espace sémantique — indépendamment de l'historique d'un utilisateur particulier.

**Validation qualitative :** Les tests sur des titres comme *Maus* ou *Harry Potter* ont produit des résultats cohérents. *Maus* trouvait *Persepolis* en tête — deux romans graphiques autobiographiques traitant de la résistance et de l'histoire.

### 4.8 Entraînement et optimisation du modèle SVD

Le modèle collaboratif retenu est **SVD biaisé** (Singular Value Decomposition), entraîné avec la bibliothèque `scikit-surprise`. Une optimisation des hyperparamètres est réalisée via GridSearchCV en validation croisée à 5 plis, en faisant varier le nombre de facteurs latents, le taux d'apprentissage et le coefficient de régularisation.

Le meilleur modèle utilise 150 facteurs latents, 30 itérations, et un coefficient de régularisation de 0,1.

### 4.9 Conception du recommandeur hybride

Le système hybride combine trois signaux :
- Le score collaboratif issu de SVD
- Le score de similarité content-based (cosinus entre profil utilisateur et embeddings livres)
- Le score de popularité IMDb

Un **alpha adaptatif** (fonction sigmoïde sur le nombre d'interactions) pondère dynamiquement SVD et content-based selon l'activité de l'utilisateur. Un **boost de genre** et un **reranking MMR** complètent le système.

### 4.10 Évaluation en ranking Top-K

Le système est évalué avec un protocole Leave-One-Out : on retire un livre de l'historique de chaque utilisateur éligible, on génère les recommandations parmi 101 candidats (1 positif + 100 négatifs aléatoires), et on mesure si le livre retiré apparaît dans le Top-10.

Les métriques calculées incluent HR@10, MRR@10, NDCG@10, Precision@10 et Recall@10.

### 4.11 Tests de significativité et analyse par segment

Le test de Wilcoxon est appliqué pour vérifier que les différences de performance entre modèles sont statistiquement significatives. Une analyse par segment (profils froids, intermédiaires, actifs) permet d'évaluer le comportement adaptatif de l'hybride.

### 4.12 Modèle auxiliaire d'interprétation avec SHAP

Un modèle Random Forest auxiliaire est entraîné pour prédire les préférences relatives des utilisateurs, et les valeurs SHAP sont calculées pour identifier les signaux les plus influents dans le processus de recommandation.

### 4.13 Sauvegarde des artefacts

Tous les composants nécessaires à l'application Flask sont sérialisés et sauvegardés dans le dossier `models/`. Cette étape marque le passage du notebook expérimental au système déployable.

---

## 5. Modèles de recommandation

### 5.1 Baseline popularité

**Ce qu'il fait :** Recommande les mêmes livres à tous les utilisateurs, en les triant par score de popularité IMDb.

**Quand il est utile :** Pour les utilisateurs totalement nouveaux (cold start), ou comme référence de comparaison.

**Forces :** Robuste, rapide, stable. Difficile à battre sur des datasets très sparses, car les livres populaires plaisent généralement à beaucoup de personnes.

**Limites :** Aucune personnalisation. Tout le monde reçoit la même liste, indépendamment de ses goûts.

---

### 5.2 Filtrage collaboratif (SVD)

**Ce qu'il fait :** Apprend des facteurs latents à partir des patterns de notation de 32 776 utilisateurs. Pour un utilisateur donné, il prédit la note qu'il attribuerait à chaque livre non encore lu, et recommande les mieux notés.

**Quand il est utile :** Pour les utilisateurs avec un historique suffisant (segment Lecteur ou Passionné).

**Forces :** Capture des affinités subtiles et non explicites — par exemple, que les fans de fantasy épique apprécient souvent certains romans de science-fiction, sans que cela ait été défini à l'avance.

**Limites :** Fragile pour les profils légers (cold start). Avec peu d'interactions, le modèle a peu appris sur l'utilisateur, et ses prédictions s'approchent de la moyenne globale.

---

### 5.3 Recommandation par contenu (Jina Embeddings v4)

**Ce qu'il fait :** Construit une représentation vectorielle de chaque livre à partir de son texte (titre, description, genre, auteur). La similarité entre livres est mesurée par la similarité cosinus entre leurs vecteurs.

**Quand il est utile :** Pour les nouveaux utilisateurs (avec des livres aimés fournis en entrée), et pour les recommandations livre-livre.

**Forces :** Ne dépend pas de l'historique d'interactions. Peut recommander des livres nouveaux ou peu évalués si leur contenu est proche du profil de l'utilisateur.

**Limites :** 89,3 % des livres ont des embeddings construits presque uniquement sur le titre. La capacité discriminante est faible pour ces livres, et les recommandations tendent à converger vers les mêmes titres populaires.

---

### 5.4 Recommandeur hybride

**Ce qu'il fait :** Combine les trois signaux précédents de façon pondérée, avec un alpha adaptatif qui donne plus de poids au collaboratif quand l'historique est riche, et plus de poids au content-based et à la popularité quand il est léger.

**Quand il est utile :** Pour tous les utilisateurs connus — c'est le mode de recommandation principal de BooksMatch.

**Forces :** Compense les faiblesses de chaque composante. Se dégrade gracieusement selon la richesse du profil.

**Limites :** Sur ce dataset, l'hybride ne dépasse pas massivement la popularité en raison de la sparsité. Il prend nettement l'avantage pour les utilisateurs actifs, mais l'amélioration est plus modeste pour les profils légers.

---

## 6. Logique hybride finale

### Formule de scoring

```
score(u, i) = α · z_CF + (1 - α) · z_CB + β · z_Pop + boost_genre
```

| Variable | Description |
|---|---|
| `z_CF` | Score SVD normalisé en z-score |
| `z_CB` | Score content-based normalisé |
| `z_Pop` | Score de popularité normalisé |
| `α` | Poids adaptatif (sigmoïde sur le nombre d'interactions) |
| `β` | Poids de la popularité (valeur optimisée = 0,50) |
| `boost_genre` | +0,15 si le genre primaire correspond au profil utilisateur |

### Alpha adaptatif

```
α(n) = 0.5 + 0.35 / (1 + exp(-0.5 × (n - 7)))
```

| Segment | Interactions | Alpha approx. | Signal dominant |
|---|---|---|---|
| Découvreur | < 5 | ≈ 0,54 | Contenu + popularité |
| Lecteur | 5–15 | ≈ 0,70 | Mix équilibré |
| Passionné | > 15 | ≈ 0,85 | SVD |

### Routage selon le contexte

| Situation | Stratégie |
|---|---|
| Utilisateur connu | Hybride adaptatif SVD + CB + Popularité |
| Nouveau lecteur avec livres aimés | CB + Popularité + boost genre |
| Visiteur sans historique | Popularité IMDb |
| Recherche de livres similaires | Similarité cosinus livre-livre |
| Filtre par genre explicite | Hybride contraint au genre sélectionné |

### MMR Reranking

Avant de retourner les recommandations, un reranking par **Maximal Marginal Relevance** est appliqué (λ = 0,5, maximum 4 livres du même genre primaire dans le Top-10). Cela garantit une diversité suffisante dans la liste finale, en évitant que plusieurs tomes d'une même série monopolisent les résultats.

---

## 7. Évaluation

### Métriques SVD

| Métrique | Valeur |
|---|---|
| RMSE (split final 80/20) | 0,9488 |
| MAE (split final) | 0,7567 |
| RMSE CV 5-fold optimisé | 0,9469 ± 0,0048 |

### Protocole de ranking

- **Utilisateurs éligibles :** 9 086
- **TOP_K :** 10
- **N négatifs par utilisateur :** 100
- **best_beta :** 0,50

### Résultats comparatifs

| Modèle | HR@10 | MRR@10 | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|---|
| Popularité | 0,2463 | **0,1101** | **0,1213** | **0,0259** | 0,2004 |
| SVD seul | 0,1960 | 0,0622 | 0,0787 | — | — |
| Contenu seul | 0,1960 | 0,0682 | 0,0828 | — | — |
| **Hybride** | **0,2464** | 0,0998 | 0,1145 | 0,0258 | **0,2011** |

### Interprétation honnête des résultats

L'hybride obtient le meilleur HR@10 et le meilleur Recall@10, mais la popularité reste supérieure sur MRR et NDCG. Ce résultat peut sembler décevant, mais il est cohérent avec la structure du dataset.

La popularité constitue une baseline extrêmement forte dans les systèmes de recommandation très sparses : les livres populaires le sont précisément parce qu'ils plaisent à beaucoup de personnes, et donc à la plupart des utilisateurs également. Quand les profils individuels sont trop légers pour que la personnalisation s'exprime, recommander ce que tout le monde aime reste une stratégie difficile à battre globalement.

L'hybride se distingue clairement de SVD seul (+0,025 en HR@10) et de contenu seul (+0,025 en HR@10). Et surtout, l'analyse par segment révèle que l'écart entre l'hybride et la popularité se creuse nettement pour les utilisateurs actifs. C'est là que SVD déploie sa valeur, et c'est exactement le comportement attendu d'un système adaptatif.

---

## 8. Interprétabilité avec SHAP

Pour comprendre pourquoi le système recommande ce qu'il recommande, un **modèle auxiliaire supervisé** (Random Forest) a été entraîné sur les recommandations générées. La variable cible est la note relative de chaque livre par rapport à la moyenne de l'utilisateur — ce qui corrige le biais de sévérité entre utilisateurs.

Les **valeurs SHAP** permettent de quantifier la contribution de chaque signal à la prédiction, cas par cas.

### Signaux les plus influents

| Rang | Signal | Interprétation |
|---|---|---|
| 1 | `cf_score` (SVD) | Signal dominant pour les profils actifs |
| 2 | `rating_mean` | Qualité perçue globale du livre |
| 3 | `pop_score` | Volume et fiabilité de l'évaluation |
| 4 | `cb_score` | Utile mais secondaire — limité par les métadonnées |

**R² du modèle auxiliaire : 0,41** — raisonnable pour un problème aussi bruité.

Ces résultats confirment la cohérence de l'architecture hybride : les signaux inclus sont bien ceux qui influencent le plus les recommandations, et aucune variable ne semble superflue.

---

## 9. Application web

BooksMatch est exposé via une application web complète, accessible depuis un navigateur sans installation.

### Architecture

- **Backend :** Flask (Python) — expose les routes API de recommandation et charge les artefacts en mémoire au démarrage
- **Frontend :** HTML5 / CSS3 / JavaScript vanilla — interface réactive construite sans framework

### Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| Recommandations personnalisées | Connexion par numéro utilisateur → Top-10 hybride adaptatif |
| Onboarding nouveau lecteur | Sélection de livres aimés → recommandations content-based + popularité |
| Cold start | Visiteur anonyme → Top populaires par genre |
| Livres similaires | Clic sur un livre → 8 livres proches par similarité cosinus |
| Filtre par genre | Navigation du catalogue avec filtre sur le genre primaire |
| Recherche | Recherche par titre ou auteur dans le catalogue complet |

### Routes API principales

| Route | Méthode | Description |
|---|---|---|
| `/api/recommend/user` | POST | Recommandations pour utilisateur connu |
| `/api/recommend/new` | POST | Recommandations pour nouveau lecteur |
| `/api/recommend/popular` | GET | Livres populaires (avec filtre genre optionnel) |
| `/api/recommend/similar` | POST | Livres similaires à un titre |
| `/api/books` | GET | Catalogue complet (filtrable) |
| `/api/users/<uid>` | GET | Profil et historique d'un utilisateur |

---

## 10. Artefacts sauvegardés

À la fin du notebook, tous les composants nécessaires à l'application sont sérialisés et sauvegardés dans le dossier `models/`.

| Fichier | Contenu |
|---|---|
| `svd_model.pkl` | Modèle SVD entraîné et optimisé |
| `book_embeddings_df.pkl` | Embeddings Jina v4 pour les 898 livres |
| `embedding_matrix.npy` | Matrice vectorielle 898 × 2048 (NumPy) |
| `books_base_df_final.pkl` | Catalogue enrichi avec scores et métadonnées |
| `user_profiles_df_final.pkl` | Profils embedding des utilisateurs |
| `train_df.pkl` | Données d'entraînement (split 80/20) |
| `test_df.pkl` | Données de test |
| `popular_books_df.pkl` | Classement des livres par popularité IMDb |
| `hybrid_config.json` | Paramètres du système hybride (best_beta, etc.) |

**Pourquoi sauvegarder ces artefacts ?** Générer les embeddings Jina et entraîner SVD sont des opérations coûteuses. En les sauvegardant, l'application Flask peut démarrer en quelques secondes sans recalculer quoi que ce soit. Cela garantit également la reproductibilité des recommandations entre deux sessions.

---

## 11. Installation et configuration

### Prérequis

- Python 3.9 ou supérieur
- Une clé API Jina (pour la génération des embeddings dans le notebook)
- Les dépendances listées dans `requirements.txt`

### Étapes d'installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-utilisateur/booksmatch.git
cd booksmatch

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API Jina (uniquement pour le notebook)
export JINA_API_KEY="votre_clé_api"    # Linux / macOS
set JINA_API_KEY="votre_clé_api"       # Windows
```

> **Note :** Si les artefacts sont déjà générés et sauvegardés dans `models/`, il n'est pas nécessaire de réexécuter le notebook pour lancer l'application Flask.

### Dépendances principales

```
flask
flask-cors
numpy
pandas
scikit-learn
scikit-surprise
requests
shap
```

---

## 12. Utilisation

### Exécuter le notebook

Ouvrez le notebook `Final_Final.ipynb` dans Jupyter ou JupyterLab et exécutez les cellules dans l'ordre. L'exécution complète génère et sauvegarde tous les artefacts dans le dossier `models/`.

```bash
jupyter notebook notebook/Final_Final.ipynb
```

### Lancer l'application Flask

```bash
python app.py
```

L'application est accessible à l'adresse `http://127.0.0.1:5000` par défaut.

### Ce que l'utilisateur peut faire

- **Se connecter** avec son numéro d'utilisateur pour recevoir des recommandations personnalisées
- **Explorer** sans compte en sélectionnant des livres aimés depuis la page *Mes préférences*
- **Parcourir** les tendances et les livres populaires depuis l'accueil
- **Rechercher** un livre par titre ou auteur
- **Découvrir** des livres similaires depuis la fiche de n'importe quel titre
- **Filtrer** le catalogue par genre
- **Consulter** son historique de lecture depuis le tableau de bord utilisateur

---

## 13. Limites identifiées

BooksMatch a été construit en toute transparence sur ses limites. Les reconnaître est essentiel pour interpréter correctement les résultats et orienter les améliorations futures.

- **Sparsité extrême :** 99,67 % de la matrice utilisateur-livre est vide. La majorité des utilisateurs ont des profils trop légers pour que la personnalisation s'exprime pleinement.

- **Faible couverture des métadonnées :** 89,3 % des livres n'ont pas de description complète. Les embeddings de ces livres sont construits presque uniquement sur le titre, ce qui limite la discrimination content-based.

- **Popularité difficile à battre :** Sur ce type de dataset, la baseline popularité est extrêmement robuste. L'hybride dépasse la popularité pour les utilisateurs actifs, mais l'amélioration reste modeste en vision globale.

- **Personnalisation limitée en cold start :** Pour un utilisateur totalement nouveau, les recommandations sont nécessairement génériques. L'onboarding par sélection de livres aimés améliore rapidement cette situation, mais la première expérience reste peu différenciée.

- **Absence de signal temporel :** Le système ne modélise pas l'évolution des goûts dans le temps. Un utilisateur dont les préférences ont changé reçoit des recommandations basées sur l'ensemble de son historique, sans pondération temporelle.

---

## 14. Pistes d'amélioration

- **Enrichissement des métadonnées :** Scraper des sources comme Open Library ou Google Books API pour compléter les descriptions manquantes, ce qui améliorerait significativement la discrimination content-based.

- **Signal implicite :** Intégrer des comportements implicites (livres ouverts mais non notés, temps passé sur une fiche) pour enrichir les profils utilisateurs au-delà des notes explicites.

- **Onboarding amélioré :** Proposer un questionnaire de démarrage plus guidé pour accélérer la construction du profil des nouveaux utilisateurs.

- **Modèles plus avancés :** Tester LightFM (hybride natif), Neural Collaborative Filtering, ou des approches de type séquentielles pour mieux capturer la dynamique temporelle des goûts.

- **Déploiement cloud :** Héberger l'application sur une plateforme comme Render ou Railway pour la rendre accessible publiquement, en gérant le stockage des artefacts lourds (≈ 300 Mo) via un service externe.

- **Évaluation en ligne et A/B testing :** Mettre en place un mécanisme de collecte de feedback utilisateur (clics, livres ajoutés à la liste de lecture) pour évaluer l'impact réel des recommandations dans un contexte d'usage réel.

---

## 15. Conclusion

BooksMatch est un projet complet, construit de bout en bout dans des conditions de données réellement difficiles. Plutôt que de chercher un modèle unique plus sophistiqué qui aurait artificiellement résolu les problèmes de sparsité, le projet a misé sur une architecture hybride robuste et transparente — une architecture qui sait quand faire confiance à chaque signal, qui se dégrade gracieusement quand l'information manque, et qui devient progressivement plus personnalisée à mesure que l'utilisateur interagit avec elle.

Les résultats sont honnêtes : l'hybride ne bat pas massivement la popularité en vision globale, mais il la surpasse nettement pour les utilisateurs actifs, et il dépasse clairement SVD seul et content seul. Ce comportement adaptatif est précisément ce qu'on attendait d'un système conçu pour des données imparfaites.

Le projet constitue une base déployable et défendable pour des travaux futurs. L'application Flask opérationnelle, les artefacts sauvegardés, la documentation complète et le rapport technique forment ensemble un livrable cohérent, reproductible et extensible.

---

*BooksMatch — IFM30542-H2026 · Groupe 7 · Collège La Cité · Ottawa · 2026*
