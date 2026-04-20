import torch
from torch.utils.data import Dataset
import pandas as pd

class MovieLensDataset(Dataset):
    def __init__(self, filepath, user2idx, movie2idx):
        df = pd.read_csv(filepath)
        self.users  = torch.tensor([user2idx[u] for u in df["user_id"].values], dtype=torch.long)
        self.movies = torch.tensor([movie2idx[m] for m in df["movie_id"].values], dtype=torch.long)
        self.ratings = torch.tensor(
            (df["rating"].values >= 4).astype(float), dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]