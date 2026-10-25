
"""Подготовка PPE-датасета с классами helmet и vest."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets/construction-ppe"
OUTPUT = ROOT / "datasets/ppe"

CLASS_MAP = {0: 0, 2: 1}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Папка уже существует: {OUTPUT}")

    for split in ("train", "val", "test"):
        images_src = SOURCE / "images" / split
        labels_src = SOURCE / "labels" / split

        images_dst = OUTPUT / "images" / split
        labels_dst = OUTPUT / "labels" / split

        # Копируем изображения
        shutil.copytree(images_src, images_dst)
        labels_dst.mkdir(parents=True)

        # Преобразуем разметку
        for label_file in labels_src.glob("*.txt"):
            new_lines = []

            for line in label_file.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if not fields:
                    continue

                class_id = int(fields[0])

                if class_id in CLASS_MAP:
                    fields[0] = str(CLASS_MAP[class_id])
                    new_lines.append(" ".join(fields))

            output_file = labels_dst / label_file.name
            output_file.write_text(
                "\n".join(new_lines),
                encoding="utf-8",
            )

        print(f"Подготовлена выборка: {split}")

    print("Датасет PPE подготовлен")


if __name__ == "__main__":
    main()
