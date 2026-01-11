import torch
import torch.nn as nn

class ConceptBottleneckModel(nn.Module):
    def __init__(self, encoder, concept_head, classifier):
        super().__init__()
        self.encoder = encoder
        self.concept_head = concept_head
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        c_hat = self.concept_head(features)
        y_hat = self.classifier(c_hat)
        return c_hat, y_hat
