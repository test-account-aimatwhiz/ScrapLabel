Project: ScrapLabeller

GPU:
- NVIDIA GeForce RTX 4070 SUPER
- VRAM: 11.99 GB

Python:
- Python 3.10.20

CUDA:
- CUDA Runtime: 12.1

PyTorch:
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- torchaudio==2.5.1+cu121

Core Libraries:
- numpy==1.26.4
- opencv-python==4.11.0.86
- supervision==0.25.1
- streamlit==1.45.1

SAM2:
- Version: 1.0
- Commit:
  2b90b9f5ceec907a1c18123530e92e794ad901a4

Checkpoint:
- sam2.1_hiera_large.pt

Config:
- sam2.1_hiera_l.yaml

Status:
✓ CUDA Working
✓ GPU Detected
✓ SAM2 Installed
✓ SAM2 CUDA Extension Compiled
✓ Model Loaded Successfully