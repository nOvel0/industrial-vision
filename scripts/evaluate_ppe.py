from ultralytics import YOLO

model = YOLO("models/best.pt")

for confidence in (0.40,):
    results = model.val(
        data="configs/dataset.yaml",
        split="test",
        conf=confidence,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        plots=True,
        project="runs/thresholds",
        name=f"conf_{confidence}",
    )

    # Матрица: строки — предсказания, столбцы — разметка.
    matrix = results.confusion_matrix.matrix

    # Класс 0 — helmet.
    tp = matrix[0, 0]
    fp = matrix[0, :].sum() - tp
    fn = matrix[:, 0].sum() - tp

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0

    print(
        f"\nHELMET | confidence={confidence:.2f} | "
        f"TP={tp:.0f} FP={fp:.0f} FN={fn:.0f} | "
        f"Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}\n"
    )