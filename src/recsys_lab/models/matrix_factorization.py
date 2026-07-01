import torch
from torch import nn

class MatrixFactorization(nn.Module):

    def __init__(self,
                 num_users : int,
                 num_items : int,
                 embedding_dim : int) -> None:
        
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    
    def forward(self,
               user_idx: torch.Tensor,
               item_idx: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embeddings(user_idx)
        item_vectors = self.item_embeddings(item_idx)

        'Dot Product Score'

        scores = (user_vectors * item_vectors).sum(dim = 1)

        return scores
    
       