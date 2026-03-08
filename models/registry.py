# models/registry.py
from .mlp import MLP
from .cnn import SimpleCNN
from .lenet5 import LeNet5
from .resnet import ResNet

MODEL_REGISTRY = {
    "mlp" : MLP,
    "cnn" : SimpleCNN,
    "lenet5" : LeNet5,
    "resnet" : ResNet
}

def get_model(name : str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    
    return MODEL_REGISTRY[name]()