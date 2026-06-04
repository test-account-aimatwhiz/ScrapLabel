import os
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOUR VERIFIED WORKING CONFIG (from debug)
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

IMAGE_PATH = "/home/matwhiz/ScrapLabeller/images/industrial.jpeg"

OUTPUT_PATH = "/home/matwhiz/ScrapLabeller/outputs/segmented9.png"

# =====================================================
# AUTO IMAGE SCALING
# =====================================================

def auto_scale_image(image_path, target_dim=1280):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    longest_side = max(h, w)

    if longest_side <= target_dim:
        return image, 1.0

    scale = target_dim / longest_side

    new_w = int(w * scale)
    new_h = int(h * scale)

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return image, scale

# =====================================================
# FILTER SMALL / HUGE MASKS
# =====================================================

def filter_masks(masks, min_area=150, max_area=300000):

    return [
        m for m in masks
        if min_area <= m["area"] <= max_area
    ]

# =====================================================
# IOU
# =====================================================

def mask_iou(mask1, mask2):

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0

    return intersection / union

# =====================================================
# REMOVE DUPLICATES
# =====================================================

def remove_duplicate_masks(masks, iou_threshold=0.50):

    keep = []

    for m in masks:
        current_mask = m["segmentation"]

        duplicate = False

        for existing in keep:

            if mask_iou(current_mask, existing["segmentation"]) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            keep.append(m)

    return keep

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading SAM2...")

model = build_sam2(
    config_file=CONFIG,
    ckpt_path=CHECKPOINT,
    device=DEVICE
)

# =====================================================
# MASK GENERATOR
# =====================================================

print("Creating Mask Generator...")

mask_generator = SAM2AutomaticMaskGenerator(
    model=model,


    points_per_side=32,
    points_per_batch=128,


    pred_iou_thresh=0.65,


    stability_score_thresh=0.72,
    stability_score_offset=0.90,


    crop_n_layers=1,
    crop_n_points_downscale_factor=2,


    crop_overlap_ratio=0.50,


    min_mask_region_area=20,


    box_nms_thresh=0.60,


    use_m2m=True
)



# =====================================================
# READ IMAGE
# =====================================================

print("Reading image...")

image, scale = auto_scale_image(IMAGE_PATH, target_dim=1280)

print(f"Processing: {image.shape[1]}x{image.shape[0]}")
print(f"Scale Factor: {scale:.3f}")

# =====================================================
# VRAM CLEANUP
# =====================================================

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

start_time = time.time()

# =====================================================
# GENERATE MASKS
# =====================================================

print("Generating masks...")

masks = mask_generator.generate(image)

print(f"\nRaw Masks: {len(masks)}")

areas = [m["area"] for m in masks]

print(f"Smallest Mask: {min(areas)} px")
print(f"Largest Mask: {max(areas)} px")
print(f"Median Area: {int(np.median(areas))}")
print(f"Mean Area: {int(np.mean(areas))}")

# =====================================================
# FILTER MASKS
# =====================================================

masks = filter_masks(masks)

print(f"After Area Filtering: {len(masks)}")

# =====================================================
# SORT BY QUALITY
# =====================================================

masks = sorted(
    masks,
    key=lambda x: (x["predicted_iou"], x["stability_score"]),
    reverse=True
)

# =====================================================
# REMOVE DUPLICATES
# =====================================================

masks = remove_duplicate_masks(masks)

print(f"After Duplicate Removal: {len(masks)}")

# =====================================================
# VRAM METRICS
# =====================================================

end_time = time.time()

if torch.cuda.is_available():

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print(f"\nPeak VRAM Allocated: {peak_allocated:.2f} GB")
    print(f"Peak VRAM Reserved: {peak_reserved:.2f} GB")

print(f"Execution Time: {end_time - start_time:.2f}s")

# =====================================================
# VISUALIZATION
# =====================================================

overlay = image.copy()
rng = np.random.default_rng(42)

for mask_data in masks:

    mask = mask_data["segmentation"]

    color = rng.integers(0, 255, size=3, dtype=np.uint8)

    overlay[mask] = (
        overlay[mask] * 0.4 + color * 0.6
    ).astype(np.uint8)

# =====================================================
# SAVE OUTPUT
# =====================================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

plt.figure(figsize=(12, 12))
plt.imshow(overlay)
plt.axis("off")

plt.savefig(
    OUTPUT_PATH,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("\n✅ Segmentation Complete")
print(f"Saved to: {OUTPUT_PATH}")