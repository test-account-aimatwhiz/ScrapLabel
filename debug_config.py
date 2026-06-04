import torch
from sam2.build_sam import build_sam2

print("SAM2 Model Loader Started...\n")

# --------------------------------------------------
# CONFIG (use the one that worked first)
# --------------------------------------------------

WORKING_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

# fallback configs (optional safety)
FALLBACK_CONFIGS = [
    WORKING_CONFIG,
    "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1_hiera_l.yaml",
    "sam2_hiera_l.yaml",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
used_config = None

print("Device:", device)
print("\nTrying to load SAM2 model...\n")

# --------------------------------------------------
# TRY LOADING MODEL
# --------------------------------------------------

for cfg in FALLBACK_CONFIGS:

    print("--------------------------------------------------")
    print("Trying config:", cfg)

    try:
        model = build_sam2(
            config_file=cfg,
            ckpt_path=CHECKPOINT,
            device=device
        )

        used_config = cfg
        print("\n✅ SUCCESS! SAM2 loaded with config:")
        print(cfg)
        break

    except Exception as e:
        print("❌ FAILED")
        print("Error:", str(e))

# --------------------------------------------------
# FINAL STATUS
# --------------------------------------------------

if model is None:
    print("\n❌ SAM2 MODEL LOADING FAILED. Check config & checkpoint paths.")
else:
    print("\n🎯 FINAL RESULT")
    print("Working Config:", used_config)
    print("SAM2 is ready to use!")