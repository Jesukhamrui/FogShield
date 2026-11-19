"""Create clear-vs-fog-vs-dehazed comparison images with detection boxes.

Each output PNG has three panels: clear | fog (detect without dehazing) | dehazed (two-stage).

Usage:
  python scripts/make_comparison.py --weights yolov8n.pt --num 6
  python scripts/make_comparison.py --weights models/yolo_fog.pt --num 6 --split val
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

COLORS = {"person": (0, 255, 255), "vehicle": (0, 255, 0)}


def draw(img, dets):
    out = img.copy()
    for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
        x1, y1, x2, y2 = map(int, box)
        c = COLORS.get(label, (255, 0, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{label} {score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return out


def titled(img, title, height=360):
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * height / h), height))
    bar = np.full((34, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return np.vstack([bar, img])


def make_panel(clear, hazy, detect, dehaze):
    """detect(img)->dets dict; dehaze(img)->img. Returns one 3-panel image."""
    enhanced = dehaze(hazy)
    n_f = len(detect(hazy)["labels"])
    d_e = detect(enhanced)
    panels = [
        titled(draw(clear, detect(clear)), "Clear"),
        titled(draw(hazy, detect(hazy)), f"Fog, no dehazing ({n_f} detections)"),
        titled(draw(enhanced, d_e), f"Dehazed + detect ({len(d_e['labels'])} detections)"),
    ]
    h = min(p.shape[0] for p in panels)
    return np.hstack([p[:h] for p in panels])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--split", choices=["all", "val"], default="all")
    ap.add_argument("--clear", default="data/clear")
    ap.add_argument("--hazed", default="data/hazed")
    ap.add_argument("--out", default="results/comparison")
    args = ap.parse_args()

    from src.dehaze import DarkChannelDehazer
    from src.detector import Detector

    with open(os.path.join(args.hazed, "fog_params.json")) as f:
        params = json.load(f)
    names = sorted(params)
    if args.split == "val":
        val = set(os.listdir(os.path.join("data", "yolo", "images", "val")))
        names = [n for n in names if n in val]
    if not names:
        raise SystemExit("No frames found.")
    step = max(1, len(names) // args.num)
    names = names[::step][: args.num]

    det, dz = Detector(args.weights), DarkChannelDehazer()
    os.makedirs(args.out, exist_ok=True)
    for n in names:
        clear = cv2.imread(os.path.join(args.clear, params[n]["source"]))
        hazy = cv2.imread(os.path.join(args.hazed, n))
        if clear is None or hazy is None:
            continue
        panel = make_panel(clear, hazy, det, dz.dehaze)
        cv2.imwrite(os.path.join(args.out, os.path.splitext(n)[0] + "_comparison.png"), panel)
    print(f"Saved {len(names)} comparison images to {args.out}/")


if __name__ == "__main__":
    main()
