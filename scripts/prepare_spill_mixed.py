"""Сборка разметки разливов и ZIP для Colab."""

import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from .prepare_spill import DATASETS, read_split


LOCAL = DATASETS / "spill_yolo_seg"
OIL = DATASETS / "oil_spill.v1i.coco-segmentation"
PIAP = DATASETS / "Liquid_detaset"
OUTPUT = DATASETS / "spill_mixed_yolo"
OIL_SELECTION = Path(__file__).with_name("spill_oil_v1_selection.txt")

# Съёмки на терракотовой плитке целиком отложены для проверки.
PIAP_VAL = set(range(55, 64)) | set(range(73, 86))
# Проверенные кадры без видимого разлива.
PIAP_BACKGROUND = {15, 17, 41, 54, 109, 111}


def source_images():
    for split in ("train", "val"):
        for image in sorted((LOCAL / "images" / split).iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = LOCAL / "labels" / split / f"{image.stem}.txt"
            yield "local", split, image, label.read_text(encoding="utf-8").splitlines()

    oil_labels = {}
    for split in ("train", "valid", "test"):
        for image, labels in read_split(OIL / split, category="oil_spill"):
            oil_labels[image.relative_to(OIL).as_posix()] = labels

    for line in OIL_SELECTION.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        labels = oil_labels[name]
        if not labels:
            raise ValueError(f"Нет масок в выбранном кадре Roboflow: {name}")
        yield "oil", "train", OIL / name, labels

    annotations = PIAP / "annotations" / "instances_default.json"
    for image, labels in read_split(PIAP / "images", "liquid", annotations):
        frame = int(image.stem.removeprefix("frame_"))
        if not labels and frame not in PIAP_BACKGROUND:
            continue
        split = "val" if frame in PIAP_VAL else "train"
        yield "piap", split, image, labels


def main():
    archive = OUTPUT.with_suffix(".zip")
    if OUTPUT.exists() or archive.exists():
        raise FileExistsError(f"Результат сборки уже существует: {OUTPUT} или {archive}")

    counts = Counter()
    backgrounds = Counter()
    with TemporaryDirectory(dir=DATASETS) as work:
        work = Path(work)
        folder = work / OUTPUT.name
        for split in ("train", "val"):
            (folder / "images" / split).mkdir(parents=True)
            (folder / "labels" / split).mkdir(parents=True)

        for source, split, image, labels in source_images():
            stem = f"{source}_{image.stem}"
            label_path = folder / "labels" / split / f"{stem}.txt"
            with label_path.open("x", encoding="utf-8") as file:
                file.writelines(f"{label}\n" for label in labels)
            target = folder / "images" / split / f"{stem}{image.suffix.lower()}"
            shutil.copy2(image, target)
            counts[source, split] += 1
            backgrounds[source, split] += not labels

        (folder / "data.yaml").write_text(
            "train: images/train\nval: images/val\nnames:\n  0: spill\n",
            encoding="utf-8",
        )

        # В архиве папка и data.yaml сохраняют свои пути.
        zipped = work / archive.name
        with ZipFile(zipped, "w", compression=ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(work))

        folder.rename(OUTPUT)
        zipped.rename(archive)

    for (source, split), count in sorted(counts.items()):
        print(f"{source} {split}: {count} кадров, из них {backgrounds[source, split]} без разлива")
    print(f"Готово: {archive}")


if __name__ == "__main__":
    main()
