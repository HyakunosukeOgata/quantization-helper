# scripts/eval_accuracy.py

import argparse
import torch
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from quantizer.ptq import dynamic_quantize

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main(args):
    print("Loading ResNet18...")
    weights = ResNet18_Weights.DEFAULT
    model_fp32 = resnet18(weights=weights)

    print("Preparing validation dataset...")
    transform = weights.transforms()
    dataset = ImageFolder(args.data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Evaluating FP32 model...")
    acc_fp32 = evaluate(model_fp32, dataloader)

    print("Applying dynamic quantization...")
    model_int8 = dynamic_quantize(model_fp32)

    print("Evaluating INT8 model...")
    acc_int8 = evaluate(model_int8, dataloader)

    print("\n=== Accuracy Report ===")
    print(f"FP32 Accuracy: {acc_fp32:.4f}")
    print(f"INT8 Accuracy: {acc_int8:.4f}")
    print(f"Accuracy Drop: {acc_fp32 - acc_int8:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to validation image folder (ImageNet format)")
    args = parser.parse_args()
    main(args)
