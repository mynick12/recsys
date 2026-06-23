from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens_100k(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "ml-100k.zip"
    extracted_dir = raw_dir / "ml-100k"

    if extracted_dir.exists():
        print(f"MovieLens 100K already exists at {extracted_dir}")
        return

    print(f"Downloading MovieLens 100K to {zip_path}")
    urlretrieve(MOVIELENS_100K_URL, zip_path)

    print(f"Extracting {zip_path}")
    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(raw_dir)

    print(f"Done. Dataset available at {extracted_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    download_movielens_100k(raw_dir)