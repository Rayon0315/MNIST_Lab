# experiments/export.py
import sys
sys.path.append('../')

from models.registry import get_model
from utils.export_onnx import export_onnx

model = get_model('mlp').to('cpu')
export_onnx(model, 'mlp')