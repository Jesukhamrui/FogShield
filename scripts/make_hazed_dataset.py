"""Create synthetic hazy frames from clear images using the atmospheric scattering model.

Each clear image gets `--per-image` hazy versions with randomly sampled fog density
(beta) and airlight. Annotation keys are kept identical so ground truth carries over.

Usage:
  python scripts/make_hazed_dataset.py --clear data/clear --out data/hazed --per-image 3
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hazing import add_fog  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--beta-range", type=float, nargs=2, default=[0.6, 1.6])
    ap.add_argument("--airlight-range", type=float, nargs=2, default=[0.8, 1.0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names = sorted(f for f in os.listdir(args.clear) if f.lower().endswith(IMG_EXTS))
    count = 0
    index = {}
    for name in names:
        img = cv2.imread(os.path.join(args.clear, name))
        if img is None:
            continue
        stem, ext = os.path.splitext(name)
        for k in range(args.per_image):
            beta = float(rng.uniform(*args.beta_range))
            A = float(rng.uniform(*args.airlight_range))
            hazy = add_fog(img, beta=beta, airlight=A)
            out_name = f"{stem}_fog{k}{ext}"
            cv2.imwrite(os.path.join(args.out, out_name), hazy)
            index[out_name] = {"source": name, "beta": beta, "airlight": A}
            count += 1
    with open(os.path.join(args.out, "fog_params.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {count} hazy frames to {args.out}")


if __name__ == "__main__":
    main()
