# utils/export_onnx.py
from pathlib import Path
import torch

ONNX_DIR = Path("../onnx")
ONNX_DIR.mkdir(exist_ok=True)


def export_onnx(
    model,
    name: str,
    input_shape=(1, 1, 28, 28),
    opset=11
):
    model.eval()

    dummy = torch.randn(*input_shape)

    path = ONNX_DIR / f"{name}.onnx"
    torch.onnx.export(
        model,
        dummy,
        path,
        dynamo = False,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
    )
    return path