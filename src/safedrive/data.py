"""Audited image loading, without a PyTorch dependency in stage 1."""
import csv
import hashlib
from pathlib import Path
from PIL import Image

SPLITS = ("train", "validation", "final")


def assign_drivers(subjects, seed, counts):
    """Stable across Python versions; never selects a split using model results."""
    if set(counts) != set(SPLITS) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Require positive train/validation/final driver counts")
    subjects = sorted(set(subjects))
    if len(subjects) != sum(counts.values()):
        raise ValueError("Driver count does not match split configuration")
    ranked = sorted(subjects, key=lambda s: (hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(), s))
    result, start = {}, 0
    for split in SPLITS:
        result[split] = sorted(ranked[start:start + counts[split]])
        start += counts[split]
    return result


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes dataset root: {relative}")
    return path


class DriverImageDataset:
    """Map-style dataset usable by a future torch DataLoader with a tensor transform.

    By default returns RGB PIL images. Stochastic transforms belong only on train.
    Final evaluation requires an explicit opt-in to discourage accidental tuning.
    """

    def __init__(self, root, manifest, split, transform=None, *, allow_final=False,
                 verify_hashes=False):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if split == "final" and not allow_final:
            raise ValueError("Final holdout is locked; use allow_final=True only for frozen evaluation")
        self.root = Path(root)
        self.transform = transform
        self.verify_hashes = verify_hashes
        with open(manifest, newline="") as stream:
            all_rows = list(csv.DictReader(stream))
        required = {"path", "img", "subject", "label", "classname", "split", "sha256"}
        if not all_rows or not required <= all_rows[0].keys():
            raise ValueError("Missing or invalid audited manifest")
        ids, driver_splits = set(), {}
        for row in all_rows:
            if row["img"] in ids:
                raise ValueError("Duplicate image ID in manifest")
            ids.add(row["img"])
            label = int(row["label"])
            if label not in range(10) or row["classname"] != f"c{label}" or row["split"] not in SPLITS:
                raise ValueError("Invalid label or split in manifest")
            previous = driver_splits.setdefault(row["subject"], row["split"])
            if previous != row["split"]:
                raise ValueError("Driver leakage in manifest")
            safe_path(self.root, row["path"])
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = safe_path(self.root, row["path"])
        if self.verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image changed after audit: {row['img']}")
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": int(row["label"]),
                "image_id": row["img"], "subject": row["subject"]}
