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