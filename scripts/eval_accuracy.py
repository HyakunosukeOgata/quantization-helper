# scripts/eval_accuracy.py

import argparse
import torch
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from quantizer.ptq import dynamic_quantize


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main(args):
    device = torch.device(args.device)

    print("Loading ResNet18...")
    weights = ResNet18_Weights.DEFAULT
    model_fp32 = resnet18(weights=weights).to(device)

    print("Preparing validation dataset...")
    transform = weights.transforms()
    dataset = ImageFolder(args.data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Evaluating FP32 model...")
    acc_fp32 = evaluate(model_fp32, dataloader, device)

    print("Applying dynamic quantization...")
    model_int8 = dynamic_quantize(model_fp32)  # already on CPU
    acc_int8 = evaluate(model_int8, dataloader, torch.device("cpu"))  # 必須用 CPU 推論

    print("\n=== Accuracy Report ===")
    print(f"FP32 Accuracy: {acc_fp32:.4f}")
    print(f"INT8 Accuracy: {acc_int8:.4f}")
    print(f"Accuracy Drop: {acc_fp32 - acc_int8:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to validation image folder (ImageNet format)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run FP32 model on (cpu or cuda)")
    args = parser.parse_args()
    main(args)
