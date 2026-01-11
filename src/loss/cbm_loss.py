import torch
import torch.nn as nn

class CBMLoss(nn.Module):
    def __init__(self, task_loss, concept_loss, lambda_):
        super().__init__()
        self.task_loss = task_loss
        self.concept_loss = concept_loss
        self.lambda_ = lambda_

    def forward(self, y_hat, y, c_hat, c):
        task = self.task_loss(y_hat, y)
        concept = self.concept_loss(c_hat, c)
        total = task + self.lambda_ * concept
        return total
