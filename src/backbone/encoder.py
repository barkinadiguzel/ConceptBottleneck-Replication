import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.net(x)
