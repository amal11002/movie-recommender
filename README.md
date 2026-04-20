# Movie Recommender — 8INF974
## Système de recommandation de films avec NeuMF + Streamlit

## Architecture du projet

```
movie-recommender/
├── data/
│   ├── ml-100k/          # Dataset brut MovieLens 100K
│   ├── train.csv         # Split entraînement (80%)
│   ├── test.csv          # Split test (20%)
│   └── movies_clean.csv  # Films nettoyés
├── src/
│   ├── download_data.py  # Téléchargement MovieLens
│   ├── explore_data.py   # Exploration et statistiques
│   └── prepare_data.py   # Nettoyage et split train/test
├── notebooks/
├── venv/
└── README.md
```

---

## Phase 1 — Setup & données 

### Dataset : MovieLens 100K
- **Source** : [grouplens.org](https://files.grouplens.org/datasets/movielens/ml-100k.zip)
- **Contenu** : 100 000 notes (1–5) par 943 utilisateurs sur 1 682 films
- **Fichiers clés** :
  - `u.data` — ratings (user_id, movie_id, rating, timestamp)
  - `u.item` — informations sur les films (id, titre, genres)
  - `u.user` — informations sur les utilisateurs (id, âge, genre, profession)

### Scripts

#### `src/download_data.py`
Télécharge et décompresse le dataset MovieLens 100K depuis le serveur officiel GroupLens dans le dossier `data/`.

#### `src/explore_data.py`
Charge les trois fichiers principaux du dataset et affiche :
- Les premières lignes de chaque table
- Les dimensions (shape)
- Le nombre d'utilisateurs et de films uniques
- La distribution des notes (moyenne, min, max)

#### `src/prepare_data.py`
Prépare les données pour l'entraînement :
- Vérifie les valeurs manquantes
- Ré-indexe les `user_id` et `movie_id` à partir de 0 (obligatoire pour les embeddings PyTorch)
- Effectue un split **80% train / 20% test** (random_state=42 pour reproductibilité)
- Sauvegarde `train.csv`, `test.csv` et `movies_clean.csv`

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/amal11002/movie-recommender.git
cd movie-recommender

# Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate 

# Installer les dépendances
pip install pandas numpy matplotlib seaborn requests scikit-learn
```

## Lancer la Phase 1

```bash
python src/download_data.py   # Télécharge les données
python src/explore_data.py    # Explore le dataset
python src/prepare_data.py    # Prépare train/test
```

---

## Phase 2 — Modèle NeuMF 

### Architecture : Neural Matrix Factorization (NeuMF)

NeuMF est une architecture de deep learning pour les systèmes de recommandation, proposée par He et al. en 2017. Elle dépasse le filtrage collaboratif classique (simple produit scalaire) en ajoutant un réseau de neurones pour capturer des relations non-linéaires entre utilisateurs et films.

Le modèle combine **deux branches parallèles** :

```
User ID ──┬── GMF Embedding ──────────────── (produit) ──┐
          │                                               ├── Linear → score
Movie ID ──┴── GMF Embedding                              │
                                                          │
User ID ──┬── MLP Embedding ──┐                           │
          │                   ├── concat → MLP layers ───┘
Movie ID ──┴── MLP Embedding ──┘
```

- Branche GMF (Generalized Matrix Factorization) : fait un produit élément par élément entre l'embedding utilisateur et l'embedding film. C'est la version "classique" mais apprise par le réseau.
- Branche MLP : concatène les deux embeddings et les passe dans plusieurs couches fully connected avec ReLU. Cela permet au modèle d'apprendre des interactions complexes et non-linéaires.
- Fusion : les sorties des deux branches sont concaténées et passées dans une dernière couche linéaire qui produit le score final (note prédite).

### Scripts

#### `src/dataset.py`
Wrapper PyTorch (Dataset) qui charge le CSV, convertit les colonnes en tenseurs, et normalise les notes entre 0 et 1.

#### `src/model.py`
Définit l'architecture NeuMF. Paramètres configurables : embedding_dim (défaut : 32) et mlp_layers (défaut : [64, 32, 16]).

#### `src/train.py`
Valide l'architecture sur 10 epochs avec les données préparées en Phase 1. Sauvegarde le modèle dans data/neumf_model.pt.

## Lancer la Phase 1

```bash
python src/train.py
```

### Résultats obtenus

| Epoch | Train Loss | Test RMSE |
|-------|-----------|-----------|
| 1     | 0.0758    | 0.1904    |
| 2     | 0.0407    | 0.1842    |
| 3     | 0.0316    | **0.1831** |
| 10    | 0.0106    | 0.2098    |

L'architecture fonctionne. Le RMSE remonte après l'epoch 3 (overfitting léger) à ce stade, la Phase 3 s'occupera d'optimiser ça.

## Phase 3 — Interface StreamliT

### Scripts

#### `src/app.py`
Interface interactive avec deux modes de recommandation :
- **Par films** : tape un titre, le système trouve l'utilisateur le plus similaire parmi les 943 profils
- **Par genres** : sélectionne un ou plusieurs genres, le système retourne les films les mieux scorés
pip install streamlit
streamlit run src/app.py