from setuptools import setup, find_packages

setup(
    name="quant-easy",
    version="0.1.0",
    packages=find_packages(), 
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "pytest>=7.0.0"
    ],
    author="Ogata.shih",
    description="A simple CLI tool for PyTorch model quantization.",
)
