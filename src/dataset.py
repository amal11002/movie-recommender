import torch
from torch.utils.data import Dataset
import pandas as pd

class MovieLensDataset(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        self.users  = torch.tensor(df["user_id"].values,  dtype=torch.long)
        self.movies = torch.tensor(df["movie_id"].values, dtype=torch.long)
        # Binaire : note >= 4 = aimé (1), sinon pas aimé (0)
        self.ratings = torch.tensor(
            (df["rating"].values >= 4).astype(float), dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]