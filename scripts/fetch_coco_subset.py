"""Download a street-scene subset of COCO val2017 with human-drawn boxes.

Picks images that contain at least one vehicle (car, bus, truck, motorcycle, bicycle),
saves them to data/clear/ and writes data/clear_labels.json (person/vehicle boxes).

Downloads ~241 MB of annotations once, then only the selected images (~160 KB each).

Usage:
  python scripts/fetch_coco_subset.py --num 700
"""
import argparse
import json
import os
import random
import urllib.request
import zipfile

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# COCO category ids -> our two classes
CLASS_OF = {1: "person", 2: "vehicle", 3: "vehicle", 4: "vehicle", 6: "vehicle", 8: "vehicle"}
MIN_AREA = 400  # ignore boxes smaller than ~20x20 px


def select_images(coco, num, seed=42, min_vehicles=1):
    """Return [(file_name, url, boxes, labels)] for up to `num` street-like images."""
    per_img = {}
    for a in coco["annotations"]:
        if a.get("iscrowd") or a["category_id"] not in CLASS_OF or a["area"] < MIN_AREA:
            continue
        x, y, w, h = a["bbox"]
        per_img.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], CLASS_OF[a["category_id"]]))
    info = {im["id"]: im for im in coco["images"]}
    chosen = []
    for iid, items in per_img.items():
        if sum(1 for _, l in items if l == "vehicle") >= min_vehicles:
            chosen.append(iid)
    chosen.sort()
    random.Random(seed).shuffle(chosen)
    out = []
    for iid in chosen[:num]:
        items = per_img[iid]
        out.append((info[iid]["file_name"], info[iid]["coco_url"],
                    [b for b, _ in items], [l for _, l in items]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=700)
    ap.add_argument("--clear", default="data/clear")
    ap.add_argument("--labels-out", default="data/clear_labels.json")
    args = ap.parse_args()

    os.makedirs(args.clear, exist_ok=True)
    os.makedirs("data/_cache", exist_ok=True)
    ann_json = "data/_cache/instances_val2017.json"
    if not os.path.exists(ann_json):
        zpath = "data/_cache/annotations.zip"
        print("Downloading COCO annotations (~241 MB), one time only...")
        urllib.request.urlretrieve(ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            with z.open("annotations/instances_val2017.json") as src, open(ann_json, "wb") as dst:
                dst.write(src.read())
        os.remove(zpath)
    with open(ann_json) as f:
        coco = json.load(f)

    picked = select_images(coco, args.num)
    labels = {}
    for i, (name, url, boxes, labs) in enumerate(picked, 1):
        dest = os.path.join(args.clear, name)
        if not os.path.exists(dest):
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:  # skip a bad download, keep going
                print(f"skip {name}: {e}")
                continue
        labels[name] = {"boxes": boxes, "labels": labs}
        if i % 50 == 0:
            print(f"{i}/{len(picked)} images")
    with open(args.labels_out, "w") as f:
        json.dump(labels, f)
    print(f"Saved {len(labels)} images to {args.clear}/ and labels to {args.labels_out}")


if __name__ == "__main__":
    main()
