import shutil
from pathlib import Path

from ultralytics import YOLO


DATASET_ROOT = Path("classification_dataset")
EXPORT_PATH = Path("models") / "custom-items-cls.pt"
FALLBACK_MODEL = "yolo11n-cls.pt"
PROJECT_DIR = "runs-yolo-cls"
RUN_NAME = "custom-items-cls-finetune"
TRAIN_TIME_HOURS = 0.67


def validate_dataset():
    train_dir = DATASET_ROOT / "train"
    val_dir = DATASET_ROOT / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "classification_dataset/train oder classification_dataset/val fehlt. "
            "Starte zuerst prepare_yolo_dataset.py."
        )

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if len(class_names) < 2:
        raise RuntimeError("Fuer das Training werden mindestens 2 Klassen benoetigt.")

    return class_names


def pick_model_source():
    if EXPORT_PATH.exists():
        return str(EXPORT_PATH)
    return FALLBACK_MODEL


def main():
    class_names = validate_dataset()
    print(f"Training startet fuer {len(class_names)} Klassen: {', '.join(class_names)}")

    model_source = pick_model_source()
    print(f"Basis-Modell: {model_source}")

    model = YOLO(model_source)
    model.train(
        data=str(DATASET_ROOT),
        epochs=999,
        time=TRAIN_TIME_HOURS,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        patience=0,
        dropout=0.10,
        fliplr=0.0,
        erasing=0.15,
        degrees=4.0,
        translate=0.04,
        scale=0.2,
        perspective=0.0005,
        auto_augment="randaugment",
        project=PROJECT_DIR,
        name=RUN_NAME,
    )

    best_path = Path(model.trainer.best)
    if not best_path.exists():
        raise FileNotFoundError(f"Bestes Modell wurde nicht gefunden: {best_path}")

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, EXPORT_PATH)

    print("Training abgeschlossen.")
    print(f"Bestes Modell: {best_path}")
    print(f"Kopiert nach: {EXPORT_PATH}")


if __name__ == "__main__":
    main()
