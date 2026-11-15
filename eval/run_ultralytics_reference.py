#!/usr/bin/env python3
"""Run the Ultralytics reference model on the fixed COCO image slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--slice", type=Path, default=Path("data/val_slice.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/val2017"))
    parser.add_argument("--out", type=Path, default=Path("results_pytorch.json"))
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.slice.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Slice must contain a non-empty list: {args.slice}")

    image_ids = [int(record["id"]) for record in records]
    image_paths = [args.data_root / str(record["file_name"]) for record in records]
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation image: {missing[0]}")

    model = YOLO(args.weights)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    print(f"[info] weights={args.weights} params={parameter_count / 1e6:.1f}M")
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        batch=1,
        half=True,
        device=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        stream=True,
        verbose=False,
    )

    detections: list[dict[str, object]] = []
    result_count = 0
    for result_count, result in enumerate(results, start=1):
        if result_count > len(image_ids):
            raise RuntimeError(f"Model returned more than {len(image_ids)} results")
        image_id = image_ids[result_count - 1]
        if result.boxes is None:
            continue
        boxes = result.boxes.xywh.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        for box, score, cls in zip(boxes, scores, classes):
            class_index = int(cls)
            center_x, center_y, width, height = (float(value) for value in box)
            detections.append({
                "image_id": image_id,
                "category_id": COCO80_TO_91[class_index],
                "bbox": [
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    width,
                    height,
                ],
                "score": float(score),
            })
    if result_count != len(image_ids):
        raise RuntimeError(f"Model returned {result_count} results for {len(image_ids)} images")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    print(f"[write] {args.out} detections={len(detections)}")


if __name__ == "__main__":
    main()
