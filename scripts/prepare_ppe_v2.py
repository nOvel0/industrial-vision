from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ROOT / "datasets/ppe"
CHV = ROOT / "datasets/chv_original"
OUTPUT = ROOT / "datasets/ppe_v2_simple"

random.seed(42)

# Создаём новый каталог. Исходные данные остаются на месте.
OUTPUT.mkdir(parents=True)

for split in ("train", "val", "test"):
    (OUTPUT / "images" / split).mkdir(parents=True)
    (OUTPUT / "labels" / split).mkdir(parents=True)

train_images = sorted(
    p for p in (TRAIN / "images/train").iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

chv_images = sorted(
    p for p in (CHV / "images").iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

# При 1132 train получаем по 242 изображения val и test.
count = round(len(train_images) * 15 / 70)

random.shuffle(chv_images)

val_images = chv_images[:count]
test_images = chv_images[count:count * 2]

# Копируем текущую обучающую выборку.
for image in train_images:
    shutil.copy2(image, OUTPUT / "images/train" / image.name)

    label = TRAIN / "labels/train" / f"{image.stem}.txt"
    shutil.copy2(label, OUTPUT / "labels/train" / label.name)

# CHV: жилет 1 → 1, все цвета касок → 0.
class_map = {"1": "1", "2": "0", "3": "0", "4": "0", "5": "0"}

for split, images in (("val", val_images), ("test", test_images)):
    for image in images:
        shutil.copy2(image, OUTPUT / "images" / split / image.name)

        label = CHV / "annotations" / f"{image.stem}.txt"
        new_lines = []

        for line in label.read_text().splitlines():
            fields = line.split()

            if fields and fields[0] in class_map:
                fields[0] = class_map[fields[0]]
                new_lines.append(" ".join(fields))

        destination = OUTPUT / "labels" / split / label.name
        destination.write_text("\n".join(new_lines), encoding="utf-8")

# Конфигурация для локального запуска.
config = (
    f"path: {OUTPUT.relative_to(ROOT).as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: helmet\n"
    "  1: vest\n"
)

(ROOT / "configs/dataset.yaml").write_text(
    config,
    encoding="utf-8",
)

print("Train:", len(train_images))
print("Val:", len(val_images))
print("Test:", len(test_images))
print("Датасет:", OUTPUT)