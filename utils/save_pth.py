# utils/save_pth.py
from pathlib import Path
import torch

# 在experiments/ 下运行
PTH_DIR = Path('../pth')
PTH_DIR.mkdir(exist_ok = True)

def save_pth(model, name: str):
    path = PTH_DIR / f'{name}.pth'
    torch.save(model.state_dict(), path)
    return path