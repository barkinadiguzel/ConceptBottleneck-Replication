import torch

def intervene(model, x, concept_idx, new_value):
    model.eval()
    with torch.no_grad():
        c_hat, _ = model(x)
        c_hat[:, concept_idx] = new_value
        y_new = model.classifier(c_hat)
    return y_new
