"""Подготовка датасета разливов для YOLO Segmentation."""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory


DATASETS = Path(__file__).resolve().parents[1] / "datasets"
SOURCE = DATASETS / "------.v10i.coco-segmentation"
OUTPUT = DATASETS / "spill_yolo_seg"

# Ламинат и уличные съёмки отложены для проверки целиком.
VAL_NUMBERS = set(range(13, 21)) | {30, 32, 34} | set(range(35, 51))


def read_split(folder, category="water", annotation_path=None):
    annotation_path = annotation_path or folder / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category["name"] for category in data["categories"]}
    labels = defaultdict(list)

    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        polygons = annotation["segmentation"]
        if categories[annotation["category_id"]] != category or len(polygons) != 1:
            raise ValueError(f"Неожиданная разметка в {image['file_name']}")

        coordinates = polygons[0]
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"Некорректный контур в {image['file_name']}")

        # YOLO хранит координаты в долях ширины и высоты.
        points = [
            value / (image["width"] if index % 2 == 0 else image["height"])
            for index, value in enumerate(coordinates)
        ]
        if not all(0 <= value <= 1 for value in points):
            raise ValueError(f"Контур выходит за границы {image['file_name']}")

        labels[image["id"]].append("0 " + " ".join(f"{value:.6f}" for value in points))

    for image in images.values():
        yield folder / image["file_name"], labels[image["id"]]


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Папка уже существует: {OUTPUT}")

    splits = {"train": [], "val": []}
    val_originals = set()

    # Для val берётся один вариант кадра, по возможности исходный.
    for source_split in ("valid", "train"):
        for image, labels in sorted(read_split(SOURCE / source_split)):
            original = image.name.split(".rf.", 1)[0]
            match = re.fullmatch(r"(\d+)_JPG", original)
            if match and int(match[1]) in VAL_NUMBERS:
                if original not in val_originals:
                    splits["val"].append((image, labels))
                    val_originals.add(original)
            else:
                splits["train"].append((image, labels))

    if not all(splits.values()):
        raise ValueError("Для обучения и проверки нужны изображения")

    with TemporaryDirectory(dir=DATASETS) as temporary_dir:
        prepared = Path(temporary_dir) / OUTPUT.name
        for split, items in splits.items():
            image_dir = prepared / "images" / split
            label_dir = prepared / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            for image, labels in items:
                shutil.copy2(image, image_dir / image.name)
                label_path = label_dir / f"{image.stem}.txt"
                with label_path.open("x", encoding="utf-8") as file:
                    file.writelines(f"{label}\n" for label in labels)

        (prepared / "data.yaml").write_text(
            "train: images/train\nval: images/val\nnames:\n  0: spill\n",
            encoding="utf-8",
        )
        prepared.rename(OUTPUT)

    for split, items in splits.items():
        print(f"{split}: {len(items)} кадров, {sum(len(labels) for _, labels in items)} масок")
    print(f"Готово: {OUTPUT}")


if __name__ == "__main__":
    main()
