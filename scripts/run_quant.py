# scripts/run_quant.py

import argparse
import os
import sys
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# 👉 Add project root to sys.path to allow correct import of quantizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantizer.ptq import dynamic_quantize
from quantizer.static import static_quantize


def get_model_size(model, path="temp_model.pt"):
    """
    Save the model temporarily and compute its size in MB. The file is deleted afterward.
    """
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return size_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_type", type=str, default="dynamic", choices=["dynamic", "static"],
                        help="Quantization type. Choose 'dynamic' or 'static'.")
    parser.add_argument("--calib_data", type=str, default=None,
                        help="Path to calibration data folder (ImageFolder format). Required for static quant.")
    parser.add_argument("--save_path", type=str, default="quantized_model.pt",
                        help="Path to save the quantized model.")
    args = parser.parse_args()

    print("Loading pretrained resnet18 model...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    print("Calculating original model size...")
    original_size = get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")

    if args.quant_type == "dynamic":
        print("Performing dynamic quantization...")
        quantized_model = dynamic_quantize(model)

    elif args.quant_type == "static":
        if not args.calib_data:
            raise ValueError("Static quantization requires --calib_data to be specified.")

        print("Preparing calibration dataset...")
        transform = weights.transforms()
        dataset = ImageFolder(args.calib_data, transform=transform)
        calib_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        print("Performing static quantization...")
        quantized_model = static_quantize(model, calib_loader)

    else:
        raise NotImplementedError(f"Unsupported quantization type: {args.quant_type}")

    quantized_size = get_model_size(quantized_model)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {(quantized_size / original_size) * 100:.2f}%")

    print(f"💾 Saving quantized model to {args.save_path} ...")
    torch.save(quantized_model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
