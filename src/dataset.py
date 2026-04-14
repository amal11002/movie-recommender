import torch
from torch.utils.data import Dataset
import pandas as pd


class MovieLensDataset(Dataset):
    def __init__(self, filepath, user2idx, movie2idx):
     

        # =========================
        # 1. Charger les données
        # =========================
        df = pd.read_csv(filepath)

        # =========================
        # 2. Appliquer mapping (IMPORTANT)
        # =========================
        df["user_id"] = df["user_id"].map(user2idx)
        df["movie_id"] = df["movie_id"].map(movie2idx)

        # Supprimer les valeurs inconnues (sécurité)
        df = df.dropna()

        # =========================
        # 3. Conversion en tenseurs
        # =========================
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie_id"].values, dtype=torch.long)

        # =========================
        # 4. Label binaire (NeuMF)
        # =========================
        self.labels = torch.tensor(
            (df["rating"].values >= 4).astype(float),
            dtype=torch.float32
        )

        # =========================
        # 5. Infos utiles
        # =========================
        self.num_users = len(user2idx)
        self.num_movies = len(movie2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.movies[idx],
            self.labels[idx]
        )