import requests
import zipfile
import os

def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    data_dir = "data"
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extract_path = os.path.join(data_dir, "ml-100k")

    os.makedirs(data_dir, exist_ok=True)

    #  Vérifie si déjà téléchargé
    if os.path.exists(extract_path):
        print("Données déjà téléchargées ")
        return

    print("Téléchargement de MovieLens 100K...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Erreur téléchargement : {e}")
        return

    print("Extraction des données")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    except zipfile.BadZipFile:
        print("Erreur : fichier ZIP corrompu")
        return

    print("Terminé  Données prêtes dans data/ml-100k")

if __name__ == "__main__":
    download_movielens()