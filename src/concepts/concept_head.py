import torch
import torch.nn as nn

class ConceptHead(nn.Module):
    def __init__(self, in_dim, concept_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, concept_dim)

    def forward(self, features):
        return self.layer(features)
