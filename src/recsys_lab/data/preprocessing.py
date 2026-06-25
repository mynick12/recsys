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