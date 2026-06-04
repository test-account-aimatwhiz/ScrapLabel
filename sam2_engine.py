import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sam2(
    points_per_side=32,
    points_per_batch=128,
    pred_iou_thresh=0.65,
    stability_score_thresh=0.75,
    stability_score_offset=0.90,
    crop_n_layers=1,
    crop_overlap_ratio=0.40,
    min_mask_region_area=50,
    box_nms_thresh=0.65,
    use_m2m=True
):

    model = build_sam2(
        config_file=CONFIG,
        ckpt_path=CHECKPOINT,
        device=DEVICE
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=2,
        crop_overlap_ratio=crop_overlap_ratio,
        min_mask_region_area=min_mask_region_area,
        box_nms_thresh=box_nms_thresh,
        use_m2m=use_m2m
    )

    return mask_generator


def generate_masks(mask_generator, image):
    return mask_generator.generate(image)


def save_crops(image, masks):
    crops = []

    for m in masks:
        mask = m["segmentation"]

        y, x = np.where(mask)

        if len(x) == 0:
            continue

        crop = image[min(y):max(y), min(x):max(x)]
        crops.append(crop)

    return crops


def create_overlay(image, masks):
    overlay = image.copy()
    rng = np.random.default_rng(42)

    for m in masks:
        mask = m["segmentation"]
        color = rng.integers(0, 255, size=3)

        overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)

    return overlay