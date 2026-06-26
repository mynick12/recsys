from pathlib import Path

import pandas as pd


def load_movielens_100k(raw_dir: Path) -> pd.DataFrame:
    """Load MovieLens 100K ratings from the raw u.data file."""
    ratings_path = raw_dir / "ml-100k" / "u.data"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Could not find {ratings_path}. "
            "Run scripts/download_movielens.py first."
        )

    interactions = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    return interactions

def encode_ids(interactions: pd.DataFrame) -> pd.DataFrame:
    "Map users/Items ID's to contiguous numbers starting from 0"
    interactions = interactions.copy()
    user_codes, _ = pd.factorize(interactions["user_id"], sort = True)
    item_codes, _ = pd.factorize(interactions["item_id"], sort = True)

    interactions["user_idx"] = user_codes
    interactions["item_idx"] = item_codes
    
    return interactions

def make_implicit_feedback(interactions: pd.DataFrame, 
                           min_rating : int | None = None) -> pd.DataFrame:
    
    if min_rating is not None:

        interactions = interactions[interactions["rating"] >= min_rating].copy()

    interactions["label"] = 1

    return interactions

def preprocessing_movielens_100k(raw_dir : Path,
                                processed_dir : Path,
                                min_rating: int | None = None ) -> Path:
    """Create a processed MovieLens 100K interaction file.

    The pipeline is:
    1. Load raw ratings.
    2. Convert ratings into implicit feedback.
    3. Encode user/item IDs into contiguous indices.
    4. Save the processed interactions as CSV.
    """

    processed_dir.mkdir(parents = True, exist_ok = True )

    interactions = load_movielens_100k(raw_dir)
    
    interactions = make_implicit_feedback(interactions, min_rating = min_rating)

    interactions = encode_ids(interactions)

    output = processed_dir/ "interactions.csv"

    interactions.to_csv(output, index = False)

    return output
