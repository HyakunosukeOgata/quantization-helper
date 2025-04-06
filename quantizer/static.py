# quantizer/static.py

import torch
import torch.quantization
from torch.quantization import fuse_modules, prepare, convert
from torch.utils.data import DataLoader
from torch.ao.quantization import MinMaxObserver, QConfig


def fuse_model(model):
    """
    Fuse Conv+BN+ReLU modules for static quantization.
    Only works for ResNet-like structures.
    """
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block in module:
                fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
    return model


def static_quantize(model, calibration_loader: DataLoader) -> torch.nn.Module:
    """
    Apply static quantization to the given model using the calibration data.

    Args:
        model: FP32 model
        calibration_loader: DataLoader providing calibration data

    Returns:
        Quantized model (INT8)
    """
    model.cpu()
    model.eval()

    model.fuse_model = lambda: fuse_model(model)  # Patch in fuse method
    qconfig = QConfig(
        activation=MinMaxObserver.with_args(quant_min=0, quant_max=255, dtype=torch.quint8),
        weight=MinMaxObserver.with_args(quant_min=-128, quant_max=127, dtype=torch.qint8)
    )
    model.qconfig = qconfig

    model.fuse_model()
    prepare(model, inplace=True)

    # Run calibration (forward pass only)
    with torch.no_grad():
        for images, _ in calibration_loader:
            model(images)

    convert(model, inplace=True)
    return model
