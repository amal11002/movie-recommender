import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os

from dataset import MovieLensDataset
from model import NeuMF

# =========================
# CONFIG
# =========================
EMBEDDING_DIM = 32
MLP_LAYERS    = [64, 32, 16]
BATCH_SIZE    = 256
EPOCHS        = 10
LR            = 0.001
THRESHOLD     = 0.5
MODEL_PATH    = "data/neumf_model.pt"

# =========================
# CHARGEMENT MAPPING (IMPORTANT)
# =========================
df_train = pd.read_csv("data/train.csv")
df_test  = pd.read_csv("data/test.csv")

# création mapping GLOBAL (cohérent train + test)
all_users = pd.concat([df_train["user_id"], df_test["user_id"]]).unique()
all_movies = pd.concat([df_train["movie_id"], df_test["movie_id"]]).unique()

user2idx = {u: i for i, u in enumerate(all_users)}
movie2idx = {m: i for i, m in enumerate(all_movies)}

num_users = len(user2idx)
num_movies = len(movie2idx)

print(f"Utilisateurs : {num_users} | Films : {num_movies}")

# =========================
# DATASET + DATALOADER
# =========================
train_dataset = MovieLensDataset("data/train.csv", user2idx, movie2idx)
test_dataset  = MovieLensDataset("data/test.csv", user2idx, movie2idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODELE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

model = NeuMF(num_users, num_movies, EMBEDDING_DIM, MLP_LAYERS).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for users, movies, ratings in loader:
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        preds = model(users, movies)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# EVAL
# =========================
def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for users, movies, ratings in loader:
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)

            preds = model(users, movies)

            all_preds.append(preds.cpu())
            all_labels.append(ratings.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    binary_preds = (all_preds >= THRESHOLD).float()

    tp = ((binary_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((binary_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((binary_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (binary_preds == all_labels).float().mean().item()

    return precision, recall, f1, accuracy

# =========================
# TRAIN LOOP
# =========================
print("\nDébut de l'entraînement...")

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader)
    precision, recall, f1, accuracy = eval_epoch(model, test_loader)

    print(
        f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | Accuracy: {accuracy:.4f}"
    )

# =========================
# SAVE MODEL
# =========================
os.makedirs("data", exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "user2idx": user2idx,
    "movie2idx": movie2idx,
    "num_users": num_users,
    "num_movies": num_movies,
    "embedding_dim": EMBEDDING_DIM,
    "mlp_layers": MLP_LAYERS,
    "threshold": THRESHOLD
}, MODEL_PATH)

print(f"\nModèle sauvegardé dans {MODEL_PATH}")