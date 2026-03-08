# experiments/train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import csv

import sys
sys.path.append(r'../')
from models.registry import get_model
from engine.train import train_epoch
from engine.evaluate import evaluate

device = 'cpu'

# 在MNIST_LAB下运行
transform = transforms.ToTensor()
train_ds = MNIST('../data', train=True, download=True, transform=transform)
test_ds = MNIST('../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

modelType = 'resnet'
model = get_model(modelType).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0049)

# 创建并打开一个CSV文件，写入模型训练结果
csv_filename = f'../result/{modelType}_result.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'accuracy', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    for epoch in range(50):
        # 进行训练
        train_epoch(train_loader, model, loss_fn, optimizer, device)

        # 评估
        metrics = evaluate(test_loader, model, loss_fn, device)
        
        # 打印当前epoch的结果
        print(f"Epoch {epoch + 1}: accuracy = {metrics['accuracy']}, loss = {metrics['loss']}")

        # 将每个epoch的结果写入CSV
        writer.writerow({'epoch': epoch + 1, 'accuracy': metrics['accuracy'], 'loss': metrics['loss']})

# 保存模型权重和导出ONNX模型
from utils.save_pth import save_pth
from utils.export_onnx import export_onnx

outputName = modelType  # 例如 'resnet'
save_pth(model, outputName)
export_onnx(model, outputName)

print(f"训练结果已保存至 {csv_filename}")