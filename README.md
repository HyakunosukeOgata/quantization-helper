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

Run the Tests
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

Sample Output
```
Original model size: 46.84 MB
Quantized model size: 45.30 MB
Inference time (FP32): 0.020785s
Inference time (INT8): 0.020783s
Speed-up: 1.00x
Prediction (FP32): 285
Prediction (INT8): 285
```

Project Structure
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
