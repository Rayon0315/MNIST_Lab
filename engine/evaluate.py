# engine/evaluate.py
import torch

def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    correct, loss_sum = 0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_sum += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    return {
        "accuracy": correct / len(dataloader.dataset),
        "loss": loss_sum / len(dataloader)
    }