# scripts/run_quant.py

import argparse
import os
import sys
import torch
from torchvision.models import resnet18

# 👉 Add project root to sys.path to allow correct import of quantizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantizer.ptq import dynamic_quantize


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
    parser.add_argument("--quant_type", type=str, default="dynamic", choices=["dynamic"],
                        help="Quantization type. Currently supports 'dynamic' only.")
    parser.add_argument("--save_path", type=str, default="quantized_model.pt",
                        help="Path to save the quantized model.")
    args = parser.parse_args()

    print("Loading pretrained resnet18 model...")
    model = resnet18(pretrained=True)

    print("Calculating original model size...")
    original_size = get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")

    if args.quant_type == "dynamic":
        print("Performing dynamic quantization...")
        print("Original fc layer:", model.fc)
        quantized_model = dynamic_quantize(model)
        print("Quantized fc layer:", quantized_model.fc)
    else:
        raise NotImplementedError(f"Quantization type not supported: {args.quant_type}")

    quantized_size = get_model_size(quantized_model)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {(quantized_size / original_size) * 100:.2f}%")

    print(f"💾 Saving quantized model to {args.save_path} ...")
    torch.save(quantized_model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
