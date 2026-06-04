import pandas as pd
from pathlib import Path


class DatasetManager:

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            pd.DataFrame(
                columns=["image_name", "label"]
            ).to_csv(self.csv_path, index=False)

    def save_label(self, image_name, label):

        df = pd.read_csv(self.csv_path)

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[image_name, label]],
                    columns=["image_name", "label"]
                )
            ],
            ignore_index=True
        )

        df.to_csv(self.csv_path, index=False)

    def get_labeled_images(self):

        df = pd.read_csv(self.csv_path)

        return set(df["image_name"].tolist())