import os

LABELS_FILE = "database/categories.txt"


def load_categories():
    os.makedirs("database", exist_ok=True)

    if not os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "w") as f:
            f.write("metal\nplastic\nwire\nbolt\nnut\n")

    with open(LABELS_FILE, "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def add_category(label):
    labels = load_categories()

    if label not in labels:
        labels.append(label)

        with open(LABELS_FILE, "w") as f:
            f.write("\n".join(labels))


def save_label(*args, **kwargs):
    pass