#!/usr/bin/env python3
"""Draw harness detections on COCO val images to make a README demo picture.

Reads COCO-format detections (the JSON written by the C++ harness), keeps boxes
above a display threshold, and saves one PNG. Pass two detection files to get a
side-by-side comparison (e.g. FP32 | FP16) of the same images.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dets", type=Path, nargs="+", required=True,
                        help="One or two COCO-format detections JSON files")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Panel titles, one per --dets (e.g. FP32 FP16)")
    parser.add_argument("--gt", type=Path, default=Path("data/instances_val2017.json"),
                        help="COCO annotations, used only for category names")
    parser.add_argument("--slice", type=Path, default=Path("data/val_slice.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/val2017"))
    parser.add_argument("--out", type=Path, default=Path("docs/images/demo.png"))
    parser.add_argument("--score-thr", type=float, default=0.35,
                        help="Display threshold (the JSON was saved at conf 0.001)")
    parser.add_argument("--num", type=int, default=3, help="Number of images (rows)")
    parser.add_argument("--ids", type=int, nargs="+", default=None,
                        help="Specific COCO image IDs; default picks clear, uncluttered images")
    parser.add_argument("--panel-width", type=int, default=640)
    return parser.parse_args()


def load_dets(path: Path, thr: float) -> dict[int, list[dict]]:
    by_image: dict[int, list[dict]] = defaultdict(list)
    for det in json.loads(path.read_text(encoding="utf-8")):
        if float(det["score"]) >= thr:
            by_image[int(det["image_id"])].append(det)
    return by_image


def pick_images(dets: dict[int, list[dict]], slice_ids: list[int], num: int) -> list[int]:
    """Deterministic: images with 3-10 confident detections, most detections first."""
    candidates = [i for i in slice_ids if 3 <= len(dets.get(i, [])) <= 10]
    candidates.sort(key=lambda i: (-len(dets[i]), i))
    chosen = candidates[:num]
    if len(chosen) < num:  # fall back to whatever has the most detections
        rest = sorted((i for i in slice_ids if i not in chosen),
                      key=lambda i: (-len(dets.get(i, [])), i))
        chosen += rest[: num - len(chosen)]
    return chosen


def color_for(category_id: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(category_id * 9973 % (2**31))
    return tuple(int(v) for v in rng.randint(60, 230, size=3))


def draw_panel(image: np.ndarray, dets: list[dict], names: dict[int, str], width: int) -> np.ndarray:
    scale = width / image.shape[1]
    panel = cv2.resize(image, (width, int(round(image.shape[0] * scale))))
    for det in dets:
        x, y, w, h = (float(v) * scale for v in det["bbox"])
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = color_for(int(det["category_id"]))
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        text = f'{names.get(int(det["category_id"]), "?")} {float(det["score"]):.2f}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - base - 2, 0)
        cv2.rectangle(panel, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(panel, text, (x1 + 2, top + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((34, panel.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, panel])


def main() -> None:
    args = parse_args()
    if len(args.dets) > 2:
        raise ValueError("Pass at most two --dets files")
    labels = args.labels or [p.stem.replace("results_", "").upper() for p in args.dets]
    if len(labels) != len(args.dets):
        raise ValueError("--labels must match the number of --dets files")

    names = {int(c["id"]): c["name"]
             for c in json.loads(args.gt.read_text(encoding="utf-8"))["categories"]}
    records = json.loads(args.slice.read_text(encoding="utf-8"))
    files = {int(r["id"]): r["file_name"] for r in records}
    all_dets = [load_dets(path, args.score_thr) for path in args.dets]

    ids = args.ids or pick_images(all_dets[-1], [int(r["id"]) for r in records], args.num)
    rows = []
    for image_id in ids:
        image = cv2.imread(str(args.data_root / files[image_id]))
        if image is None:
            raise FileNotFoundError(f"Could not read image {files[image_id]}")
        panels = [draw_panel(image, dets.get(image_id, []), names, args.panel_width)
                  for dets in all_dets]
        rows.append(np.hstack(panels))

    width = rows[0].shape[1]
    rows = [r if r.shape[1] == width else cv2.resize(r, (width, int(r.shape[0] * width / r.shape[1])))
            for r in rows]
    panel_w = args.panel_width
    header = np.hstack([add_title(np.zeros((1, panel_w, 3), np.uint8), t)[:34] for t in labels])
    out = np.vstack([header] + rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out)
    print(f"[write] {args.out} images={ids} score_thr={args.score_thr} size={out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
