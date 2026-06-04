import os
import pandas as pd
from datetime import datetime

LABEL_FILE = "database/labels.csv"


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