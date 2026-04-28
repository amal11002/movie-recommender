import streamlit as st
import torch
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import NeuMF

# CHARGEMENT

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
    return movies[["movie_id","title","genres"]]


# RECOMMANDATION

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

# INTERFACE 
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Movie Recommender")
st.markdown(
    "Système de recommandation de films "
)
st.info(
    "Dataset MovieLens 100K (1995-1998) — les films proposés sont des classiques des années 1970 à 1998."
)

model, user2idx, movie2idx, threshold = load_model()
movies_df = load_movies()

st.markdown("---")
st.subheader("Sélectionnez des films que vous appréciez")
st.caption("Commencez à taper un titre — les suggestions apparaissent automatiquement.")

all_titles = movies_df["title"].tolist()
selected_titles = st.multiselect(
    label="Films sélectionnés :",
    options=all_titles,
    placeholder="Rechercher un film..."
)
liked_movie_ids = movies_df[movies_df["title"].isin(selected_titles)]["movie_id"].tolist()

st.markdown("---")
top_k = st.slider("Nombre de recommandations", min_value=5, max_value=20, value=10)

if st.button("Générer les recommandations", disabled=len(liked_movie_ids) == 0):
    with st.spinner("Recherche du profil le plus similaire parmi 943 utilisateurs..."):
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