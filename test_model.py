import torch
from sam2.build_sam import build_sam2

model = build_sam2(
    config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
    ckpt_path="checkpoints/sam2.1_hiera_large.pt",
    device="cuda"
)

print("Model Loaded")
print("GPU:", torch.cuda.get_device_name(0))