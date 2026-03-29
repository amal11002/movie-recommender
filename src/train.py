import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os

from dataset import MovieLensDataset
from model import NeuMF

# Config 
EMBEDDING_DIM = 32
MLP_LAYERS    = [64, 32, 16]
BATCH_SIZE    = 256
EPOCHS        = 10
LR            = 0.001
THRESHOLD     = 0.5   # seuil pour décider aimé / pas aimé
MODEL_PATH    = "data/neumf_model.pt"

#  Données
train_dataset = MovieLensDataset("data/train.csv")
test_dataset  = MovieLensDataset("data/test.csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

df_train = pd.read_csv("data/train.csv")
df_test  = pd.read_csv("data/test.csv")
num_users  = max(df_train["user_id"].max(), df_test["user_id"].max()) + 1
num_movies = max(df_train["movie_id"].max(), df_test["movie_id"].max()) + 1

print(f"Utilisateurs : {num_users} | Films : {num_movies}")

#  Modèle 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

model     = NeuMF(num_users, num_movies, EMBEDDING_DIM, MLP_LAYERS).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#  Fonctions 
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for users, movies, ratings in loader:
        users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
        optimizer.zero_grad()
        preds = model(users, movies)
        loss  = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for users, movies, ratings in loader:
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            preds = model(users, movies)
            all_preds.append(preds)
            all_labels.append(ratings)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Appliquer le seuil
    binary_preds = (all_preds >= THRESHOLD).float()

    tp = ((binary_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((binary_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((binary_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (binary_preds == all_labels).float().mean().item()

    return precision, recall, f1, accuracy

#  Boucle principale 
print("\nDébut de l'entraînement...")
for epoch in range(1, EPOCHS + 1):
    train_loss            = train_epoch(model, train_loader, optimizer, criterion)
    precision, recall, f1, accuracy = eval_epoch(model, test_loader)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | "
          f"Précision: {precision:.4f} | Rappel: {recall:.4f} | "
          f"F1: {f1:.4f} | Accuracy: {accuracy:.4f}")

#  Sauvegarde 
os.makedirs("data", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "num_users":        num_users,
    "num_movies":       num_movies,
    "embedding_dim":    EMBEDDING_DIM,
    "mlp_layers":       MLP_LAYERS,
    "threshold":        THRESHOLD,
}, MODEL_PATH)

print(f"\nModèle sauvegardé dans {MODEL_PATH}")