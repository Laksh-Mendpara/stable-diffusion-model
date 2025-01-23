# Stable Diffusion Model (Class-Conditioned DDPM)

This repository implements a class-conditioned Denoising Diffusion Probabilistic Model (DDPM) from scratch using the MNIST dataset. Additionally, it includes training and sampling utilities for a Vector Quantized Variational Autoencoder (VQVAE) and the DDPM.

---

## Repository Structure

```
.
├── config
│   └── mnist_class_cond.yaml       # Configuration file with hyperparameters and paths
├── data                            # Directory for storing dataset files
├── dataset
│   ├── __init__.py
│   └── MNIST_dataset.py            # Dataset utility for MNIST
├── LICENSE                         # License file
├── models
│   ├── blocks.py                   # Neural network building blocks
│   ├── discriminator.py            # Discriminator model
│   ├── __init__.py
│   ├── lpips.py                    # Perceptual loss (LPIPS) implementation
│   ├── unet_class_cond.py          # Class-conditioned UNet for DDPM
│   ├── vqvae.py                    # VQVAE implementation
│   └── weights                     # Directory to store model weights
├── README.md                       # Documentation (this file)
├── requirements.txt                # Required Python dependencies
├── scheduler
│   ├── __init__.py
│   └── linear_noise_scheduler.py   # Linear noise scheduler for diffusion
├── tools
│   ├── infer_vqvae.py              # Inference script for VQVAE
│   ├── __init__.py
│   ├── sample_ddpm.py              # Sampling script for DDPM
│   ├── train_ddpm.py               # Training script for DDPM
│   └── train_vqvae.py              # Training script for VQVAE
└── utils
    ├── config_utils.py             # Configuration utilities
    ├── diffusion_utils.py          # Utilities for diffusion processes
    └── __init__.py
```

---

## Setup Instructions

### Environment Setup
1. Install the `uv` package manager:
   ```bash
   pip install uv
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv <PATH OF VIRTUAL ENV> --python 3.11
   source activate <PATH OF VIRTUAL ENV>/bin/active
   ```
3. Install all dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

### Download LPIPS Weights
Download lpips weights by opening this link in browser(dont use cURL or wget) https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth and downloading the raw file. Place the downloaded weights file in ```models/weights/v0.1/vgg.pth```

---

## Training and Inference

### 1. Train VQVAE
To train the Vector Quantized Variational Autoencoder (VQVAE), run:
```bash
python -m tools.train_vqvae
```

### 2. Train Diffusion Model
To train the class-conditioned DDPM, run:
```bash
python -m tools.train_ddpm
```

### 3. Sample from Diffusion Model
To generate samples using the trained DDPM, run:
```bash
python -m tools.sample_ddpm
```

---

## Configuration
All hyperparameters and paths required for training and inference are specified in the configuration file:
```
./config/mnist_class_cond.yaml
```
---

## License
This project is licensed under the terms specified in the `LICENSE` file.

---

For any questions or contributions, feel free to create an issue or submit a pull request!

