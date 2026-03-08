# experiments/train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import sys
sys.path.append(r'../')
from models.registry import get_model
from engine.train import train_epoch
from engine.evaluate import evaluate

device = 'cpu'

# 在MNIST_LAB下运行
transform = transforms.ToTensor()
train_ds = MNIST('../data', train = True, download = True, transform = transform)
test_ds = MNIST('../data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_ds, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_ds, batch_size = 64)

modelType = 'resnet'
model = get_model(modelType).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0049)

for epoch in range(10):
    train_epoch(train_loader, model, loss_fn, optimizer, device)
    metrics = evaluate(test_loader, model, loss_fn, device)

    print(epoch + 1, metrics)

from utils.save_pth import save_pth
from utils.export_onnx import export_onnx

outputName = "resnet"
save_pth(model, outputName)
export_onnx(model, outputName)