import json
import random
import re
import shutil
import ssl
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path


DATASET_URL = "https://hook.eu1.make.celonis.com/2dkavnxbe1o4rns7k75r1ej9yqflfa7x"
ROOT = Path(__file__).resolve().parent
SOURCE_IMAGES_DIR = ROOT / "Bilder"
CLASSIFICATION_ROOT = ROOT / "classification_dataset"
TRAIN_DIR = CLASSIFICATION_ROOT / "train"
VAL_DIR = CLASSIFICATION_ROOT / "val"
ITEMS_JSON_PATH = ROOT / "yolo_dataset" / "items.json"
MAPPING_PATH = ROOT / "classification_item_mapping.json"
SUMMARY_PATH = CLASSIFICATION_ROOT / "dataset_summary.json"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VAL_RATIO = 0.2
RANDOM_SEED = 42
IGNORED_CLASS_NAMES = {"Unsicher"}


def normalize_text(value):
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_text(value))
    return slug.strip("-") or "item"


def fetch_items():
    with urllib.request.urlopen(
        DATASET_URL,
        context=ssl._create_unverified_context(),
        timeout=30,
    ) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    items = payload.get("value", [])
    ITEMS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    ITEMS_JSON_PATH.write_text(
        json.dumps(items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return items


def reset_directory(directory):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def list_class_directories():
    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Bilder-Ordner fehlt: {SOURCE_IMAGES_DIR}")

    directories = [
        path for path in SOURCE_IMAGES_DIR.iterdir()
        if path.is_dir() and path.name not in IGNORED_CLASS_NAMES and not path.name.startswith("_")
    ]
    if not directories:
        raise RuntimeError("Im Bilder-Ordner wurden keine Klassen-Unterordner gefunden.")
    return sorted(directories, key=lambda path: path.name.lower())


def list_image_files(directory):
    files = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.name.lower())


def split_files(files):
    shuffled = list(files)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    if len(shuffled) <= 1:
        return shuffled, []

    val_count = max(1, round(len(shuffled) * VAL_RATIO))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    return train_files, val_files


def copy_split(class_name, files, destination_root):
    destination_dir = destination_root / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    class_slug = slugify(class_name)

    copied = []
    for index, source_path in enumerate(files, start=1):
        target_name = f"{class_slug}_{index:03d}{source_path.suffix.lower()}"
        target_path = destination_dir / target_name
        shutil.copy2(source_path, target_path)
        copied.append(target_path.name)
    return copied


def score_candidate(class_name, item):
    class_name_normalized = normalize_text(class_name)
    item_code = normalize_text(item.get("ItemCode"))
    item_name = normalize_text(item.get("ItemName"))

    if class_name_normalized in {item_code, item_name}:
        return 1.0

    combined = " ".join(part for part in (item_code, item_name) if part)
    if not combined:
        return 0.0

    return SequenceMatcher(None, class_name_normalized, combined).ratio()


def build_mapping_template(class_names, items):
    mapping = {}
    for class_name in class_names:
        ranked = sorted(
            items,
            key=lambda item: score_candidate(class_name, item),
            reverse=True,
        )
        candidates = []
        for item in ranked[:3]:
            score = score_candidate(class_name, item)
            if score < 0.6:
                continue
            candidates.append(
                {
                    "item_code": item.get("ItemCode"),
                    "item_name": item.get("ItemName"),
                    "score": round(score, 3),
                }
            )

        resolved = next(
            (
                item for item in items
                if normalize_text(item.get("ItemCode")) == normalize_text(class_name)
                or normalize_text(item.get("ItemName")) == normalize_text(class_name)
            ),
            None,
        )

        mapping[class_name] = {
            "item_code": resolved.get("ItemCode") if resolved else None,
            "item_name": resolved.get("ItemName") if resolved else None,
            "status": "ready" if resolved else "needs_mapping",
            "candidates": candidates,
        }

    MAPPING_PATH.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_dataset():
    items = fetch_items()
    reset_directory(CLASSIFICATION_ROOT)

    class_directories = list_class_directories()
    summary = {
        "dataset_root": str(CLASSIFICATION_ROOT),
        "classes": [],
    }

    class_names = []
    for class_directory in class_directories:
        class_name = class_directory.name
        image_files = list_image_files(class_directory)
        if len(image_files) < 2:
            raise RuntimeError(
                f"Klasse '{class_name}' hat zu wenige Bilder ({len(image_files)}). Mindestens 2 werden benoetigt."
            )

        train_files, val_files = split_files(image_files)
        copied_train = copy_split(class_name, train_files, TRAIN_DIR)
        copied_val = copy_split(class_name, val_files, VAL_DIR)

        summary["classes"].append(
            {
                "class_name": class_name,
                "source_count": len(image_files),
                "train_count": len(copied_train),
                "val_count": len(copied_val),
            }
        )
        class_names.append(class_name)

    build_mapping_template(class_names, items)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Klassifikations-Datensatz vorbereitet: {CLASSIFICATION_ROOT}")
    for item in summary["classes"]:
        print(
            f"- {item['class_name']}: "
            f"{item['train_count']} train / {item['val_count']} val "
            f"(gesamt {item['source_count']})"
        )
    print(f"Mapping-Vorlage: {MAPPING_PATH}")


if __name__ == "__main__":
    prepare_dataset()
