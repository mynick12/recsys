import pandas as pd

def leave_last_two_out_split(interactions: pd.DataFrame,
                             user_col : str = "user_idx",
                             time_col : 
                             str = "timestamp") -> tuple[pd.DataFrame,
                                                                   pd.DataFrame,
                                                                   pd.DataFrame]:
    # We put the users and times that they watched a movies in order so we can use past
    # reviews as training data for the future time prediction (exactly as is expected)
    # for a recommendation system
    
    interactions = interactions.sort_values([user_col, time_col]).reset_index(drop = True).copy()

    counts_per_user = interactions.groupby(user_col).size()

    if(counts_per_user < 3).any():
        raise ValueError("For this split to work we need at least 3 reviews from the user")
    
    train_parts = []
    val_parts = []
    test_parts = []

    for _, user_interactions in interactions.groupby(user_col, sort = False):
        train_parts.append(user_interactions.iloc[:-2])
        val_parts.append(user_interactions.iloc[[-2]])
        test_parts.append(user_interactions.iloc[[-1]])

    train = pd.concat(train_parts, ignore_index= True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    return train, val, test

    