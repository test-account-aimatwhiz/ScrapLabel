import os
import json
import cv2
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

LABEL_FILE = "database/labels.csv"
DATASET_DIR = "dataset"
SESSIONS_DIR = "database/sessions"


def init_storage():
    os.makedirs("database/crops", exist_ok=True)

    if not os.path.exists(LABEL_FILE):
        df = pd.DataFrame(columns=[
            "image_name",
            "object_id",
            "label",
            "timestamp",
            "crop_path"
        ])
        df.to_csv(LABEL_FILE, index=False)


def save_to_db(image_name, object_id, label, crop_path):
    init_storage()

    df = pd.read_csv(LABEL_FILE)

    df.loc[len(df)] = [
        image_name,
        object_id,
        label,
        datetime.now().isoformat(),
        crop_path
    ]

    df.to_csv(LABEL_FILE, index=False)


def save_yolo_annotation(image_name, image, class_id, label, mask, object_id, categories):
    """Save YOLO annotation, permanent mask, and update labels.json."""
    h, w = image.shape[:2]

    # Extract contour from binary mask
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return

    # Use largest contour, simplify slightly
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return

    # Normalize to [0, 1]
    pts_norm = pts / np.array([w, h], dtype=np.float32)
    pts_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts_norm)
    bbox_str = f"{class_id} {pts_str}"

    # Directory setup
    img_dir = os.path.join(DATASET_DIR, "images", "train")
    lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    mask_dir = os.path.join(DATASET_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Save full image once
    img_path = os.path.join(img_dir, image_name)
    if not os.path.exists(img_path):
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Save permanent mask as .npy
    mask_filename = f"{object_id}_mask.npy"
    mask_path = os.path.join(mask_dir, mask_filename)
    np.save(mask_path, mask)

    # Append YOLO annotation line
    lbl_name = os.path.splitext(image_name)[0] + ".txt"
    lbl_path = os.path.join(lbl_dir, lbl_name)
    with open(lbl_path, "a") as f:
        f.write(bbox_str + "\n")

    # Update dataset.yaml
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    yaml_data = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val": "images/train",
        "nc": len(categories),
        "names": categories,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # Update labels.json with spec-compliant entry
    json_path = os.path.join(DATASET_DIR, "labels.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            records = json.load(f)
    else:
        records = []

    records.append({
        "filename": image_name,
        "image_path": img_path.replace("\\", "/"),
        "mask_path": mask_path.replace("\\", "/"),
        "bounding_box": bbox_str,
        "label": label,
    })

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)


# =========================
# SESSION PERSISTENCE
# =========================

def save_session(image_name, image, crops):
    """Save full detection session to disk immediately after SAM2 runs."""
    session_dir = os.path.join(SESSIONS_DIR, os.path.splitext(image_name)[0])
    os.makedirs(session_dir, exist_ok=True)

    # Save full image
    cv2.imwrite(
        os.path.join(session_dir, "image.png"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    # Save each crop + mask
    meta = {"image_name": image_name, "total": len(crops), "objects": []}
    for i, crop_data in enumerate(crops):
        crop_path = os.path.join(session_dir, f"crop_{i}.png")
        mask_path = os.path.join(session_dir, f"mask_{i}.npy")

        cv2.imwrite(crop_path, cv2.cvtColor(crop_data["crop"], cv2.COLOR_RGB2BGR))
        np.save(mask_path, crop_data["mask"])

        meta["objects"].append({
            "index": i,
            "crop": crop_path,
            "mask": mask_path,
            "bbox": crop_data["bbox"],
            "labeled": False
        })

    with open(os.path.join(session_dir, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return session_dir


def load_session(image_name):
    """Load a saved session from disk. Returns (image, crops, meta) or None."""
    session_dir = os.path.join(SESSIONS_DIR, os.path.splitext(image_name)[0])
    meta_path = os.path.join(session_dir, "session.json")

    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    image = cv2.imread(os.path.join(session_dir, "image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    crops = []
    for obj in meta["objects"]:
        crop = cv2.imread(obj["crop"])
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mask = np.load(obj["mask"])
        crops.append({"crop": crop, "mask": mask, "bbox": obj["bbox"]})

    return image, crops, meta


def mark_session_object_labeled(image_name, idx):
    """Mark a single object as labeled in the session file."""
    session_dir = os.path.join(SESSIONS_DIR, os.path.splitext(image_name)[0])
    meta_path = os.path.join(session_dir, "session.json")

    if not os.path.exists(meta_path):
        return

    with open(meta_path) as f:
        meta = json.load(f)

    meta["objects"][idx]["labeled"] = True

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def delete_session(image_name):
    """Remove session folder after all objects are labeled."""
    import shutil
    session_dir = os.path.join(SESSIONS_DIR, os.path.splitext(image_name)[0])
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)


def list_pending_sessions():
    """Return list of image names with incomplete sessions."""
    if not os.path.exists(SESSIONS_DIR):
        return []

    pending = []
    for name in os.listdir(SESSIONS_DIR):
        meta_path = os.path.join(SESSIONS_DIR, name, "session.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            unlabeled = sum(1 for o in meta["objects"] if not o["labeled"])
            if unlabeled > 0:
                pending.append({"image_name": meta["image_name"], "unlabeled": unlabeled})
    return pending