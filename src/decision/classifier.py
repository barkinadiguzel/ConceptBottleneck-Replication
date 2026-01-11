import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, concept_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(concept_dim, out_dim)

    def forward(self, concepts):
        return self.layer(concepts)
