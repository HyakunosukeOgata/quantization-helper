import os
import time
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from quantizer.ptq import dynamic_quantize


def get_model_size(model, path="temp_test_model.pt"):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return size_mb


def measure_inference_time(model, input_tensor, repeats=30):
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            _ = model(input_tensor)

        start = time.time()
        for _ in range(repeats):
            _ = model(input_tensor)
        end = time.time()

    return (end - start) / repeats


def test_dynamic_quantization():
    # Load pretrained model (new API with weights)
    weights = ResNet18_Weights.DEFAULT
    model_fp32 = resnet18(weights=weights)
    model_fp32.eval()

    # Calculate original model size
    original_size = get_model_size(model_fp32)

    # Apply dynamic quantization
    model_int8 = dynamic_quantize(model_fp32)

    # Check model is not None
    assert model_int8 is not None, "Quantized model is None"

    # Quantized model should be smaller
    quantized_size = get_model_size(model_int8)
    assert quantized_size < original_size, "Quantized model is not smaller than original"

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")

    # Dummy input for benchmarking
    dummy_input = torch.randn(1, 3, 224, 224)
    time_fp32 = measure_inference_time(model_fp32, dummy_input)
    time_int8 = measure_inference_time(model_int8, dummy_input)

    print(f"Inference time (FP32): {time_fp32:.6f}s")
    print(f"Inference time (INT8): {time_int8:.6f}s")
    print(f"Speed-up: {(time_fp32 / time_int8):.2f}x")

    # Accuracy check on real image
    image_path = "tests/sample_cat.jpg"
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img = Image.open(image_path).convert("RGB")

    preprocess = weights.transforms()

    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        pred_fp32 = torch.argmax(model_fp32(input_tensor), dim=1)
        pred_int8 = torch.argmax(model_int8(input_tensor), dim=1)

    print(f"Prediction (FP32): {pred_fp32.item()}")
    print(f"Prediction (INT8): {pred_int8.item()}")

    assert pred_fp32.item() == pred_int8.item(), "Top-1 predictions do not match"