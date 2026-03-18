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

### 3. Prepare Dataset

The project expects datasets inside the `dataset/` directory.

### Downloaded Datasets

- Office-31  
	Reference link: `https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#office-31`  
	Local path: `dataset/office31/`  

- Office-Home  
	Reference link: `https://www.hemanthdv.org/officeHomeDataset.html`  
	Local path: `dataset/OfficeHomeDataset/`  

- PACS  
	Reference link: `https://www.kaggle.com/datasets/ma3ple/pacs-dataset`  
	Local path: `dataset/pacs/`  

- MNIST (via torchvision)  
	Reference link: `https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html`  
	Local path: `dataset/MNIST/`  

- SVHN (via torchvision)  
	Reference link: `https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html`  
	Local path: `dataset/SVHN/`  

- USPS (via torchvision)  
	Reference link: `https://pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html`  
	Local path: `dataset/USPS/`  

### Download MNIST, SVHN, USPS from PyTorch

Run:

```bash
python scripts/download_torch_datasets.py
```

This script downloads train/test splits into dataset-specific folders:

- `dataset/MNIST/`
- `dataset/SVHN/`
- `dataset/USPS/`

### Verify All Datasets

Expected directory tree:

```text
dataset/
├── office31/
│   ├── amazon/
│   ├── dslr/
│   └── webcam/
├── OfficeHomeDataset/
│   ├── Art/
│   ├── Clipart/
│   ├── Product/
│   └── Real World/
├── pacs/
│   ├── art_painting/
│   ├── cartoon/
│   ├── photo/
│   └── sketch/
├── MNIST/
├── SVHN/
└── USPS/
```

After downloading or placing all datasets, verify they are loaded correctly:

```bash
python scripts/visualize_dataset.py
```

## Running Experiments

All configuration will be controlled in `config.yaml`. This repository will compare image style transfer in the following space:

- `spatial`: standard CycleGAN in pixel space
- `spectral`: Spectral CycleGAN where generated outputs are adapted in frequency space by translating low-frequency style while preserving high-frequency source content

### Dataset Pairs

The default config includes these 5 source -> target pairs:

- MNIST -> USPS
- SVHN -> MNIST
- Amazon -> Webcam (Office-31)
- Art -> Real-World (Office-Home)
- Photo -> Sketch (PACS)

### Run All Training Experiments

From repository root:

```bash
python scripts/run_image_style_transfer.py --config config.yaml
```

Outputs are saved in:

- `out/experiments/<pair_name>/spatial/`
- `out/experiments/<pair_name>/spectral/`

### Export Qualitative Comparison Figures

After training completes, export side-by-side qualitative plots:

```bash
python scripts/export_qualitative_comparison.py --config config.yaml --num-samples 8
```

This creates one image per pair in `out/qualitative/` with rows:

1. source image
2. CycleGAN (spatial) translation
3. Spectral CycleGAN translation
4. target-domain reference image