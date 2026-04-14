import streamlit as st
import torch
import pandas as pd
import requests

from model import NeuMF

# =========================
# LOAD DATA
# =========================
movies = pd.read_csv("data/movies_clean.csv")

# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load("data/neumf_model.pt", map_location="cpu")

user2idx = checkpoint["user2idx"]
movie2idx = checkpoint["movie2idx"]
idx2movie = {v: k for k, v in movie2idx.items()}

model = NeuMF(
    num_users=len(user2idx),
    num_movies=len(movie2idx),
    embedding_dim=checkpoint["embedding_dim"],
    mlp_layers=checkpoint["mlp_layers"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# TMDB API (posters)
# =========================
API_KEY = "TON_TMDB_API_KEY"

def get_movie_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
        res = requests.get(url).json()
        poster_path = res["results"][0]["poster_path"]
        return "https://image.tmdb.org/t/p/w500" + poster_path
    except:
        return None

# =========================
# UI
# =========================
st.title("🎬 Movie Recommender System (NeuMF)")

st.write("Choisis tes films préférés 👇")

selected_movies = st.multiselect(
    "Films aimés",
    movies["title"].values
)

# =========================
# RECOMMENDATION
# =========================
if st.button("Recommander"):

    if len(selected_movies) == 0:
        st.warning("Sélectionne au moins un film")
    else:

        # user fictif (version simple)
        user_id = list(user2idx.values())[0]
        user_tensor = torch.tensor([user_id])

        scores = []

        for movie_idx in movie2idx.values():
            movie_tensor = torch.tensor([movie_idx])

            with torch.no_grad():
                score = model(user_tensor, movie_tensor).item()

            scores.append((movie_idx, score))

        # Top 10
        top_movies = sorted(scores, key=lambda x: x[1], reverse=True)[:10]

        st.subheader("Top recommandations")

        cols = st.columns(2)

        for i, (movie_idx, score) in enumerate(top_movies):

            movie_id = idx2movie[movie_idx]
            title = movies[movies["movie_id"] == movie_id]["title"].values[0]

            poster = get_movie_poster(title)

            with cols[i % 2]:

                if poster:
                    st.image(poster, width=200)

                st.write(f"🎥 **{title}**")
                st.write(f"⭐ Score: {score:.3f}")
                st.progress(min(score, 1.0))