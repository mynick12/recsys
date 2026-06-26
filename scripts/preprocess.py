from pathlib import Path

import pandas as pd

from recsys_lab.data.preprocessing import preprocessing_movielens_100k
from recsys_lab.data.splitting import leave_last_two_out_split


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    interactions_path = preprocessing_movielens_100k(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        min_rating=None,
    )

    interactions = pd.read_csv(interactions_path)

    train, val, test = leave_last_two_out_split(interactions)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved interactions to {interactions_path}")
    print(f"Saved train split to {train_path}")
    print(f"Saved validation split to {val_path}")
    print(f"Saved test split to {test_path}")

    print()
    print(f"interactions: {interactions.shape}")
    print(f"train:        {train.shape}")
    print(f"val:          {val.shape}")
    print(f"test:         {test.shape}")