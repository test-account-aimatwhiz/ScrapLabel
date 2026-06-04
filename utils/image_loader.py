from pathlib import Path


def load_object_images(folder):

    folder = Path(folder)

    exts = {".jpg", ".jpeg", ".png"}

    files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in exts
    ]

    files.sort()

    return files