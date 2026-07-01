from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    'DataSet For user-item interaction'

    def __init__(self, interaction_path : Path,
                 user_col : str = "user_idx",
                 item_col : str = "item_idx",
                 label_col : str = "label") -> None:
        
        self.interactions = pd.read_csv(interaction_path)

        self.user_ids = torch.tensor(self.interactions[user_col].values, 
                                     dtype = torch.long)
        
        self.item_ids = torch.tensor(self.interactions[item_col].values,
                                     dtype = torch.long)
        
        self.labels = torch.tensor(self.interactions[label_col].values,
                                     dtype = torch.float32)
        
    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, index : int) -> dict[str, torch.Tensor]:
        return  {
        "user_idx": self.user_ids[index],
        "item_idx": self.item_ids[index],
        "label": self.labels[index],
    }

    
        
