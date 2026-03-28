import requests
import zipfile
import os

def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    os.makedirs("data", exist_ok=True)
    
    print("Téléchargement de MovieLens 100K")
    response = requests.get(url)
    
    zip_path = "data/ml-100k.zip"
    with open(zip_path, "wb") as f:
        f.write(response.content)
    
    print("Extraire les données")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data/")
    
    print("Terminé ! Données dans data")

if __name__ == "__main__":
    download_movielens()