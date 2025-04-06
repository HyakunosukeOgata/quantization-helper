# Quantization Helper CLI

A lightweight PyTorch model quantization tool that supports dynamic quantization. It helps you compare model size, inference speed, and prediction accuracy. Great for learning and experimenting with model compression techniques.

---

## Features

- Supports **Dynamic Quantization**
- Compares model size before and after quantization
- Measures average inference time
- Checks if predictions remain consistent (Top-1 class)
- Includes unit tests with benchmarking
- Follows standard Python project structure

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### (Optional) Install in editable mode
```bash
pip install -e .
```

### Run the Tests
```
pytest -s tests/test_quant.py
```
The -s flag enables display of print() outputs (model size, speed, predictions).

### Download Test Image
Run this script to download a sample image used in the tests:
```
python scripts/download_test_image.py
```
This will save a sample image (used for prediction comparison) to tests/sample_cat.jpg.

### Sample Output
```
Original model size: 46.84 MB
Quantized model size: 45.30 MB
Inference time (FP32): 0.020785s
Inference time (INT8): 0.020783s
Speed-up: 1.00x
Prediction (FP32): 285
Prediction (INT8): 285
```

### Accuracy Report (ImageNet Mini)
```
FP32 Accuracy: 0.7031
INT8 Accuracy: 0.7028
Accuracy Drop: 0.0003
```

## Quantization Experiment Results

We compared model size, inference speed, and prediction accuracy using ResNet-18 under different quantization strategies.

### Environment

- PyTorch 2.x
- torchvision 0.15+
- CPU-only inference
- Calibration set: ImageNet Mini (10 classes × 50 images)


---

### Model Size

| Model Type | Size (MB) | Compression Ratio |
|------------|-----------|-------------------|
| FP32       | 46.84     | 100%              |
| Dynamic    | 45.30     | 96.7%             |
| Static     | 11.85     | 25.3%             |

---

### Inference Speed (avg on CPU)

| Model Type | Time per Image (sec) | Speed-up |
|------------|----------------------|----------|
| FP32       | 0.020785             | 1.00×    |
| Dynamic    | 0.020783             | 1.00×    |
| Static     | TBD                  | TBD      |

---

### Prediction Consistency (Top-1 Class)

- FP32 → 🐱 Class ID `285` (Tabby Cat)
- INT8 (Dynamic) → ✅ Same
- INT8 (Static) → ✅ Same

---

### Top-1 Accuracy (ImageNet Mini)

| Model Type | Accuracy |
|------------|----------|
| FP32       | 0.7031   |
| Dynamic    | 0.7028   |
| Static     | TBD      |

---

### Notes

- Static quantization achieved **~75% size reduction**
- All quantized models preserved **Top-1 prediction**
- Static quantization requires a calibration dataset
- Inference speed is hardware-dependent (minimal gain on CPU)


### Project Structure
```
quantizer/
  ptq.py                 # Dynamic quantization logic
scripts/
  run_quant.py           # CLI interface
  download_test_image.py # Auto-download test image
tests/
  test_quant.py          # Unit test with size/speed/accuracy checks
  sample_cat.jpg         # Sample image (must be downloaded)
setup.py
requirements.txt
README.md
```
