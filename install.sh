#!/bin/bash

python3.10 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip setuptools wheel

pip install \
torch==2.5.1+cu121 \
torchvision==0.20.1+cu121 \
torchaudio==2.5.1+cu121 \
--index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

pip install -v --no-build-isolation \
git+https://github.com/facebookresearch/sam2.git