# AI6126 - Image-to-Image Translation and Input Space Unsupervised Domain Adaptation

This repository contains my implementation for Project I (I2I Translation and Input Space UDA) in NTU AI6126 Advanced Computer Vision. The project studies image-to-image translation for domain shift mitigation by comparing pixel-space CycleGAN and spectral-space CycleGAN (low-frequency band translation), then benchmarks input-space UDA methods including Source-only, CycleGAN-based training, spectral CycleGAN-based training, CyCADA, and FDA across multiple source->target pairs (MNIST->USPS, SVHN->MNIST, Amazon->Webcam, Art->Real-World, Photo->Sketch).

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Ardacandra/ai6126_i2i_translation_and_input_space_uda.git
cd ai6126_i2i_translation_and_input_space_uda
```

### 2. Set Up the Conda Environment

```bash
# Create a new conda environment with Python 3.9
conda create -n ai6126_i2i_translation_and_input_space_uda python=3.9 -y

# Activate the environment
conda activate ai6126_i2i_translation_and_input_space_uda

# Install PyTorch with CUDA 12.1 support for GPU acceleration
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install all other project dependencies from requirements.txt
pip install -r requirements.txt
```