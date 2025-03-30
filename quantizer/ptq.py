# quantizer/ptq.py

import torch

def dynamic_quantize(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply dynamic quantization to the given model.

    Args:
        model: The PyTorch model to be quantized (recommended to be CPU-only).

    Returns:
        A quantized model (torch.nn.Module).
    """
    model.cpu()
    model.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,                             # Only quantize nn.Linear layers
        {torch.nn.Linear},
        dtype=torch.qint8                 # Use 8-bit integer quantization
    )
    return quantized_model
