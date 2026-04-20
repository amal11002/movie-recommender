# import streamlit as st
# import torch
# import pandas as pd
# import sys
# import os

# sys.path.append(os.path.dirname(__file__))
# from model import NeuMF

# # =========================
# # CHARGEMENT
# # =========================
# @st.cache_resource
# def load_model():
#     checkpoint = torch.load("data/neumf_model.pt", map_location="cpu", weights_only=False)
#     model = NeuMF(
#         num_users     = checkpoint["num_users"],
#         num_movies    = checkpoint["num_movies"],
#         embedding_dim = checkpoint["embedding_dim"],
#         mlp_layers    = checkpoint["mlp_layers"]
#     )
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     return model, checkpoint["user2idx"], checkpoint["movie2idx"], checkpoint["threshold"]

# @st.cache_data
# def load_movies():
#     movies = pd.read_csv(
#         "data/ml-100k/u.item", sep="|", encoding="latin-1",
#         names=["movie_id","title","release_date","video_date","url",
#                "unknown","Action","Adventure","Animation","Children",
#                "Comedy","Crime","Documentary","Drama","Fantasy",
#                "FilmNoir","Horror","Musical","Mystery","Romance",
#                "SciFi","Thriller","War","Western"]
#     )
#     genre_cols = ["Action","Adventure","Animation","Children","Comedy","Crime",
#                   "Documentary","Drama","Fantasy","FilmNoir","Horror","Musical",
#                   "Mystery","Romance","SciFi","Thriller","War","Western"]
#     movies["genres"] = movies[genre_cols].apply(
#         lambda row: ", ".join([g for g in genre_cols if row[g] == 1]), axis=1
#     )
#     return movies[["movie_id","title","genres"]]

# # =========================
# # RECOMMANDATION
# # =========================
# def get_recommendations(model, user2idx, movie2idx, threshold, liked_movie_ids, movies_df, top_k=10):
#     idx2movie = {v: k for k, v in movie2idx.items()}

#     all_scores = []
#     with torch.no_grad():
#         for user_id_orig, user_idx in list(user2idx.items()):
#             liked_indices = [movie2idx[m] for m in liked_movie_ids if m in movie2idx]
#             if not liked_indices:
#                 continue
#             user_t = torch.tensor([user_idx] * len(liked_indices), dtype=torch.long)
#             movie_t = torch.tensor(liked_indices, dtype=torch.long)
#             liked_scores = torch.sigmoid(model(user_t, movie_t)).mean().item()
#             all_scores.append((user_id_orig, user_idx, liked_scores))

#     _, best_user_idx, _ = max(all_scores, key=lambda x: x[2])

#     movie_indices = list(movie2idx.values())
#     user_tensor  = torch.tensor([best_user_idx] * len(movie_indices), dtype=torch.long)
#     movie_tensor = torch.tensor(movie_indices, dtype=torch.long)

#     with torch.no_grad():
#         scores = torch.sigmoid(model(user_tensor, movie_tensor)).numpy()

#     results = []
#     for idx, score in zip(movie_indices, scores):
#         orig_movie_id = idx2movie[idx]
#         if orig_movie_id not in liked_movie_ids:
#             results.append({"movie_id": orig_movie_id, "score": float(score)})

#     results_df = pd.DataFrame(results)
#     results_df = results_df.merge(movies_df, on="movie_id")
#     results_df = results_df.sort_values("score", ascending=False).head(top_k)
#     return results_df

# # =========================
# # INTERFACE
# # =========================
# st.set_page_config(page_title="Movie Recommender", layout="wide")

# st.title("Movie Recommender")
# st.markdown(
#     "Système de recommandation de films "
# )
# st.info(
#     "Ce système est basé sur le dataset MovieLens 100K (1995-1998). "
#     "Les films proposés sont des classiques des années 1970 à 1998."
# )

# model, user2idx, movie2idx, threshold = load_model()
# movies_df = load_movies()

# # =========================
# # SELECTION FILMS
# # =========================
# st.markdown("---")
# st.subheader("Sélectionnez des films que vous appréciez")
# st.caption("Commencez à taper un titre — les suggestions apparaissent automatiquement.")

# all_titles = movies_df["title"].tolist()

# selected_titles = st.multiselect(
#     label="Films sélectionnés :",
#     options=all_titles,
#     placeholder="Rechercher un film..."
# )

# liked_movie_ids = movies_df[movies_df["title"].isin(selected_titles)]["movie_id"].tolist()

# # =========================
# # RECOMMANDATIONS
# # =========================
# st.markdown("---")

# top_k = st.slider("Nombre de recommandations", min_value=5, max_value=20, value=10)

# if st.button("Générer les recommandations", disabled=len(liked_movie_ids) == 0):
#     with st.spinner("Calcul en cours..."):
#         recs = get_recommendations(model, user2idx, movie2idx, threshold, liked_movie_ids, movies_df, top_k)

#     st.subheader("Recommandations")

#     header_col1, header_col2, header_col3 = st.columns([4, 3, 1])
#     with header_col1:
#         st.markdown("**Titre**")
#     with header_col2:
#         st.markdown("**Genres**")
#     with header_col3:
#         st.markdown("**Score**")

#     st.markdown("---")

#     for _, row in recs.iterrows():
#         col1, col2, col3 = st.columns([4, 3, 1])
#         with col1:
#             st.markdown(f"{row['title']}")
#         with col2:
#             st.markdown(f"{row['genres'] if row['genres'] else '—'}")
#         with col3:
#             st.markdown(f"**{int(row['score'] * 100)}%**")
#         st.markdown("---")

# elif len(liked_movie_ids) == 0:
#     st.caption("Sélectionnez au moins un film pour générer des recommandations.")

import streamlit as st
import torch
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import NeuMF

# =========================
# CHARGEMENT
# =========================
@st.cache_resource
def load_model():
    checkpoint = torch.load("data/neumf_model.pt", map_location="cpu", weights_only=False)
    model = NeuMF(
        num_users     = checkpoint["num_users"],
        num_movies    = checkpoint["num_movies"],
        embedding_dim = checkpoint["embedding_dim"],
        mlp_layers    = checkpoint["mlp_layers"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["user2idx"], checkpoint["movie2idx"], checkpoint["threshold"]

@st.cache_data
def load_movies():
    movies = pd.read_csv(
        "data/ml-100k/u.item", sep="|", encoding="latin-1",
        names=["movie_id","title","release_date","video_date","url",
               "unknown","Action","Adventure","Animation","Children",
               "Comedy","Crime","Documentary","Drama","Fantasy",
               "FilmNoir","Horror","Musical","Mystery","Romance",
               "SciFi","Thriller","War","Western"]
    )
    genre_cols = ["Action","Adventure","Animation","Children","Comedy","Crime",
                  "Documentary","Drama","Fantasy","FilmNoir","Horror","Musical",
                  "Mystery","Romance","SciFi","Thriller","War","Western"]
    movies["genres"] = movies[genre_cols].apply(
        lambda row: ", ".join([g for g in genre_cols if row[g] == 1]), axis=1
    )
    return movies[["movie_id","title","genres"]], genre_cols

# =========================
# RECOMMANDATION PAR FILMS
# =========================
def get_recs_by_films(model, user2idx, movie2idx, liked_movie_ids, movies_df, top_k=10):
    idx2movie = {v: k for k, v in movie2idx.items()}

    all_scores = []
    with torch.no_grad():
        for user_id_orig, user_idx in list(user2idx.items()):
            liked_indices = [movie2idx[m] for m in liked_movie_ids if m in movie2idx]
            if not liked_indices:
                continue
            user_t  = torch.tensor([user_idx] * len(liked_indices), dtype=torch.long)
            movie_t = torch.tensor(liked_indices, dtype=torch.long)
            liked_scores = torch.sigmoid(model(user_t, movie_t)).mean().item()
            all_scores.append((user_idx, liked_scores))

    _, best_user_idx = max(all_scores, key=lambda x: x[1])

    movie_indices = list(movie2idx.values())
    with torch.no_grad():
        scores = torch.sigmoid(model(
            torch.tensor([best_user_idx] * len(movie_indices), dtype=torch.long),
            torch.tensor(movie_indices, dtype=torch.long)
        )).numpy()

    results = []
    for idx, score in zip(movie_indices, scores):
        orig_id = idx2movie[idx]
        if orig_id not in liked_movie_ids:
            results.append({"movie_id": orig_id, "score": float(score)})

    results_df = pd.DataFrame(results).merge(movies_df, on="movie_id")
    return results_df.sort_values("score", ascending=False).head(top_k)

# =========================
# RECOMMANDATION PAR GENRES
# =========================
def get_recs_by_genres(model, user2idx, movie2idx, selected_genres, movies_df, top_k=10):
    idx2movie = {v: k for k, v in movie2idx.items()}

    # Filtrer les films qui correspondent aux genres sélectionnés
    genre_filter = movies_df["genres"].apply(
        lambda g: any(genre in g for genre in selected_genres)
    )
    filtered_movies = movies_df[genre_filter]

    if filtered_movies.empty:
        return pd.DataFrame()

    filtered_ids = filtered_movies["movie_id"].tolist()
    filtered_indices = [movie2idx[m] for m in filtered_ids if m in movie2idx]

    # Calculer le score moyen sur tous les utilisateurs pour ces films
    all_scores = []
    with torch.no_grad():
        for user_idx in list(user2idx.values())[:100]:
            user_t  = torch.tensor([user_idx] * len(filtered_indices), dtype=torch.long)
            movie_t = torch.tensor(filtered_indices, dtype=torch.long)
            scores  = torch.sigmoid(model(user_t, movie_t)).numpy()
            all_scores.append(scores)

    import numpy as np
    mean_scores = np.mean(all_scores, axis=0)

    results = []
    for idx, score in zip(filtered_indices, mean_scores):
        orig_id = idx2movie[idx]
        results.append({"movie_id": orig_id, "score": float(score)})

    results_df = pd.DataFrame(results).merge(movies_df, on="movie_id")
    return results_df.sort_values("score", ascending=False).head(top_k)

# =========================
# INTERFACE
# =========================
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Movie Recommender")
st.markdown(
    "Système de recommandation basé sur l'architecture **NeuMF** "
    "(Neural Matrix Factorization — He et al., 2017)."
)
st.info(
    "Dataset MovieLens 100K (1995-1998) — les films proposés sont des classiques des années 1970 à 1998."
)

model, user2idx, movie2idx, threshold = load_model()
movies_df, genre_cols = load_movies()

st.markdown("---")

# =========================
# CHOIX DU MODE
# =========================
mode = st.radio(
    "Comment souhaitez-vous obtenir des recommandations ?",
    options=["Par films que j'aime", "Par genres préférés"],
    horizontal=True
)

st.markdown("---")
top_k = st.slider("Nombre de recommandations", min_value=5, max_value=20, value=10)

# =========================
# MODE 1 — PAR FILMS
# =========================
if mode == "Par films que j'aime":
    st.subheader("Sélectionnez des films que vous appréciez")
    st.caption("Commencez à taper un titre — les suggestions apparaissent automatiquement.")

    all_titles = movies_df["title"].tolist()
    selected_titles = st.multiselect(
        label="Films sélectionnés :",
        options=all_titles,
        placeholder="Rechercher un film..."
    )
    liked_movie_ids = movies_df[movies_df["title"].isin(selected_titles)]["movie_id"].tolist()

    if st.button("Générer les recommandations", disabled=len(liked_movie_ids) == 0):
        with st.spinner("Recherche de l'utilisateur le plus similaire parmi 943 profils..."):
            recs = get_recs_by_films(model, user2idx, movie2idx, liked_movie_ids, movies_df, top_k)
        st.subheader("Recommandations")
        st.markdown("---")
        for _, row in recs.iterrows():
            col1, col2 = st.columns([5, 3])
            with col1:
                st.markdown(f"{row['title']}")
            with col2:
                st.markdown(f"{row['genres'] if row['genres'] else '—'}")
            st.markdown("---")
    elif len(liked_movie_ids) == 0:
        st.caption("Sélectionnez au moins un film pour continuer.")

# =========================
# MODE 2 — PAR GENRES
# =========================
else:
    st.subheader("Sélectionnez vos genres préférés")

    selected_genres = st.multiselect(
        label="Genres :",
        options=genre_cols,
        placeholder="Choisir un ou plusieurs genres..."
    )

    if st.button("Générer les recommandations", disabled=len(selected_genres) == 0):
        with st.spinner("Calcul des meilleurs films par genre..."):
            recs = get_recs_by_genres(model, user2idx, movie2idx, selected_genres, movies_df, top_k)

        if recs.empty:
            st.warning("Aucun film trouvé pour ces genres.")
        else:
            st.subheader("Recommandations")
            st.markdown("---")
            for _, row in recs.iterrows():
                col1, col2 = st.columns([5, 3])
                with col1:
                    st.markdown(f"{row['title']}")
                with col2:
                    st.markdown(f"{row['genres'] if row['genres'] else '—'}")
                st.markdown("---")
    elif len(selected_genres) == 0:
        st.caption("Sélectionnez au moins un genre pour continuer.")