import pandas as pd

# Chargement des ratings
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Chargement des films
movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# Chargement des utilisateurs
users = pd.read_csv(
    "data/ml-100k/u.user",
    sep="|",
    names=["user_id", "age", "gender", "occupation", "zip_code"]
)

print(" RATINGS")
print(ratings.head())
print(f"\nShape : {ratings.shape}")
print(f"Notes uniques : {ratings['rating'].unique()}")
print(f"Nb utilisateurs : {ratings['user_id'].nunique()}")
print(f"Nb films : {ratings['movie_id'].nunique()}")

print("\n FILMS ")
print(movies.head())
print(f"Shape : {movies.shape}")

print("\n UTILISATEURS ")
print(users.head())
print(f"Shape : {users.shape}")

print("\n STATISTIQUES RATINGS ")
print(ratings['rating'].describe())