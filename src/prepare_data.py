import pandas as pd
import numpy as np

# Chargement
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# Vérification valeurs manquantes
print("Valeurs manquantes ratings :", ratings.isnull().sum().sum())
print("Valeurs manquantes movies :", movies.isnull().sum().sum())

# Re-indexer user_id et movie_id à partir de 0 (nécessaire pour PyTorch)
ratings["user_id"] = ratings["user_id"] - 1
ratings["movie_id"] = ratings["movie_id"] - 1

# Split train/test (80/20)
from sklearn.model_selection import train_test_split

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

print(f"\nTrain : {train.shape}")
print(f"Test  : {test.shape}")

# Sauvegarder
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
movies.to_csv("data/movies_clean.csv", index=False)

print("\nFichiers sauvegardés dans data/")
print(f"Nb utilisateurs : {ratings['user_id'].nunique()}")
print(f"Nb films        : {ratings['movie_id'].nunique()}")