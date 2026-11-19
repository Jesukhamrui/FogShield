"""Create data/annotations.json, a YOLO-format dataset and data/fog.yaml from clear images.

Labels come from running a detector on the CLEAR images (pseudo-labels). Fog does not move
objects, so each label is copied to every hazed version of that image.

IMPORTANT: pseudo-labels measure agreement with the clear-image detector, not true accuracy.
For official results use human labels (Foggy Cityscapes / RESIDE-OTS annotations) instead.

Usage:
  python scripts/make_labels.py --labels data/clear_labels.json   # human labels (preferred)
  python scripts/make_labels.py --weights yolov8n.pt              # pseudo-labels
"""
import argparse
import json
import os
import random
import shutil
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CLASSES = {"person": 0, "vehicle": 1}


def build(detect, clear_dir, hazed_dir, out_ann, yolo_dir, yaml_path, val_frac=0.2, seed=42,
          given_labels=None):
    """detect(img_bgr) -> {'boxes','scores','labels'}, or pass given_labels (human labels) instead.
    Returns (n_hazed, n_with_objects)."""
    with open(os.path.join(hazed_dir, "fog_params.json")) as f:
        fog_params = json.load(f)

    sources = sorted({v["source"] for v in fog_params.values()})
    random.Random(seed).shuffle(sources)
    n_val = max(1, int(len(sources) * val_frac)) if len(sources) > 1 else 0
    val_set = set(sources[:n_val])  # split by SOURCE image so val never sees train scenes

    labels = {}
    for src in sources:
        img = cv2.imread(os.path.join(clear_dir, src))
        if img is None:
            continue
        if given_labels is not None:
            d = given_labels.get(src, {"boxes": [], "labels": []})
        else:
            d = detect(img)
        labels[src] = {"boxes": [list(map(float, b)) for b in d["boxes"]],
                       "labels": list(d["labels"]), "size": img.shape[:2]}

    annotations, with_obj = {}, 0
    for split in ("train", "val"):
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)

    for name, info in fog_params.items():
        lab = labels.get(info["source"])
        if lab is None or not lab["labels"]:
            continue  # skip frames with no objects: nothing to train or score on
        annotations[name] = {"boxes": lab["boxes"], "labels": lab["labels"]}
        with_obj += 1
        split = "val" if info["source"] in val_set else "train"
        shutil.copy(os.path.join(hazed_dir, name), os.path.join(yolo_dir, "images", split, name))
        h, w = lab["size"]
        lines = []
        for (x1, y1, x2, y2), c in zip(lab["boxes"], lab["labels"]):
            lines.append(f"{CLASSES[c]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        stem = os.path.splitext(name)[0]
        with open(os.path.join(yolo_dir, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    with open(out_ann, "w") as f:
        json.dump(annotations, f)
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(yolo_dir)}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: person\n  1: vehicle\n")
    return len(fog_params), with_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--labels", default=None,
                    help="JSON of human labels (e.g. data/clear_labels.json); skips pseudo-labelling")
    ap.add_argument("--clear", default="data/clear")
    ap.add_argument("--hazed", default="data/hazed")
    args = ap.parse_args()

    given, det = None, None
    if args.labels:
        with open(args.labels) as f:
            given = json.load(f)
    else:
        from src.detector import Detector
        det = Detector(args.weights, conf=args.conf)
    total, kept = build(det, args.clear, args.hazed, "data/annotations.json",
                        "data/yolo", "data/fog.yaml", given_labels=given)
    print(f"{kept} of {total} hazed frames contain vehicles/people and were labelled.")
    if kept < 2000:
        print(f"Note: the objective asks for at least 2,000 frames; you have {kept}. "
              "Add more clear photos with cars/people and regenerate the fog.")


if __name__ == "__main__":
    main()
