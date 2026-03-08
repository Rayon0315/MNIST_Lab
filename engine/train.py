# engine/train.py
import torch

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()