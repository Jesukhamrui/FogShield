"""Evaluate baseline (detect on hazed) vs two-stage (dehaze -> detect) on a hazy dataset.

Expected layout:
  data/clear/<name>.png      clear reference frames
  data/hazed/<name>.png      hazy frames (same file names as the annotations)
  data/annotations.json      {"<name>.png": {"boxes": [[x1,y1,x2,y2],...], "labels": ["vehicle",...]}}

Hazy frames without annotations are skipped. Reports:
  * mAP@0.5 on hazed frames: baseline vs two-stage
  * mAP@0.5 on clear frames (reference ceiling for the detector)
  * PSNR / SSIM of dehazed vs clear and of hazed vs clear
  * end-to-end latency (mean, median, p95)
"""
import argparse
import json
import os
import statistics
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dehaze import build_dehazer  # noqa: E402
from src.detector import Detector  # noqa: E402
from src.metrics import map50, psnr, ssim  # noqa: E402
from src.pipeline import BaselinePipeline, TwoStagePipeline  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def percentile(xs, p):
    return float(np.percentile(xs, p)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--split", choices=["all", "val"], default="all",
                    help="'val' scores only held-out images from data/yolo/images/val "
                         "(required when evaluating the fine-tuned model)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    with open(paths["annotations"]) as f:
        annotations = json.load(f)

    dz = cfg["dehazer"]
    dehazer_kwargs = ({"model_path": dz["onnx_path"], "input_size": dz["input_size"]}
                      if dz["name"] == "onnx"
                      else {"patch": dz["patch"], "omega": dz["omega"], "t0": dz["t0"]})
    dehazer = build_dehazer(dz["name"], **dehazer_kwargs)

    dc = cfg["detector"]
    detector = Detector(dc["weights"], conf=dc["conf"], imgsz=dc["imgsz"], device=dc["device"])

    baseline = BaselinePipeline(detector)
    two_stage = TwoStagePipeline(dehazer, detector)

    params_path = os.path.join(paths["hazed_dir"], "fog_params.json")
    fog_params = json.load(open(params_path)) if os.path.exists(params_path) else {}

    names = sorted(n for n in annotations if os.path.exists(os.path.join(paths["hazed_dir"], n)))
    if args.split == "val":
        val_dir = os.path.join("data", "yolo", "images", "val")
        if not os.path.isdir(val_dir):
            raise SystemExit(f"{val_dir} not found. Run scripts/make_labels.py first.")
        val_names = set(os.listdir(val_dir))
        names = [n for n in names if n in val_names]
    if args.max_frames:
        names = names[: args.max_frames]
    if not names:
        raise SystemExit("No annotated hazed frames found. Check data layout and annotations.json.")
    print(f"Evaluating {len(names)} hazed frames")

    # Warm-up: the first model call is slow; run once so it is not charged to the baseline.
    warm = cv2.imread(os.path.join(paths["hazed_dir"], names[0]))
    baseline.run(warm)
    two_stage.run(warm)

    gts = [{"boxes": annotations[n]["boxes"], "labels": annotations[n]["labels"]} for n in names]
    preds_base, preds_two, preds_clear = [], [], []
    psnr_dh, ssim_dh, psnr_hz, ssim_hz = [], [], [], []
    lat_two, lat_base, lat_dehaze = [], [], []

    for n in names:
        hazy = cv2.imread(os.path.join(paths["hazed_dir"], n))
        clear_name = fog_params.get(n, {}).get("source", n)  # road_fog0.jpg -> road.jpg
        clear_path = os.path.join(paths["clear_dir"], clear_name)
        clear = cv2.imread(clear_path) if os.path.exists(clear_path) else None

        rb = baseline.run(hazy)
        rt = two_stage.run(hazy)
        preds_base.append(rb["detections"])
        preds_two.append(rt["detections"])
        lat_base.append(rb["timings_ms"]["total"])
        lat_two.append(rt["timings_ms"]["total"])
        lat_dehaze.append(rt["timings_ms"]["dehaze"])

        if clear is not None:
            clear_det = detector(clear)
            preds_clear.append(clear_det)
            psnr_dh.append(psnr(rt["enhanced"], clear))
            ssim_dh.append(ssim(rt["enhanced"], clear))
            psnr_hz.append(psnr(hazy, clear))
            ssim_hz.append(ssim(hazy, clear))
        else:
            preds_clear.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": []})

    map_base, per_base = map50(preds_base, gts)
    map_two, per_two = map50(preds_two, gts)
    map_clear, per_clear = map50(preds_clear, gts)

    report = {
        "frames": len(names),
        "mAP50": {
            "clear_reference": map_clear,
            "hazed_baseline_no_dehaze": map_base,
            "hazed_two_stage": map_two,
            "per_class_baseline": per_base,
            "per_class_two_stage": per_two,
            "per_class_clear": per_clear,
        },
        "image_quality": {
            "psnr_dehazed_vs_clear": float(np.mean(psnr_dh)) if psnr_dh else None,
            "ssim_dehazed_vs_clear": float(np.mean(ssim_dh)) if ssim_dh else None,
            "psnr_hazed_vs_clear": float(np.mean(psnr_hz)) if psnr_hz else None,
            "ssim_hazed_vs_clear": float(np.mean(ssim_hz)) if ssim_hz else None,
        },
        "latency_ms": {
            "baseline_mean": statistics.mean(lat_base),
            "two_stage_mean": statistics.mean(lat_two),
            "two_stage_median": statistics.median(lat_two),
            "two_stage_p95": percentile(lat_two, 95),
            "dehaze_mean": statistics.mean(lat_dehaze),
        },
    }

    os.makedirs(paths["results_dir"], exist_ok=True)
    with open(os.path.join(paths["results_dir"], "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, os.path.join(paths["results_dir"], "report.md"))
    print(json.dumps({k: report[k] for k in ("mAP50", "latency_ms")}, indent=2))
    print(f"Report written to {paths['results_dir']}/report.md")


def write_markdown(r, path):
    m, q, l = r["mAP50"], r["image_quality"], r["latency_ms"]
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [
        "# Fog Detection Results",
        "",
        f"Frames evaluated: {r['frames']}",
        "",
        "## Detection (mAP@0.5)",
        "",
        "| Setting | mAP@0.5 |",
        "|---|---|",
        f"| Clear frames (reference) | {m['clear_reference']*100:.2f}% |",
        f"| Hazed, detection without dehazing (mandatory baseline) | {m['hazed_baseline_no_dehaze']*100:.2f}% |",
        f"| Hazed, two-stage dehaze + detect | {m['hazed_two_stage']*100:.2f}% |",
        "",
        "## Image quality (dehazed vs clear reference)",
        "",
        "| Metric | Hazed input | Dehazed output |",
        "|---|---|---|",
        f"| PSNR (dB) | {fmt(q['psnr_hazed_vs_clear'])} | {fmt(q['psnr_dehazed_vs_clear'])} |",
        f"| SSIM | {fmt(q['ssim_hazed_vs_clear'])} | {fmt(q['ssim_dehazed_vs_clear'])} |",
        "",
        "## Latency (ms per frame)",
        "",
        "| Pipeline | Mean | Median | p95 |",
        "|---|---|---|---|",
        f"| Baseline (detect only) | {l['baseline_mean']:.1f} | - | - |",
        f"| Two-stage | {l['two_stage_mean']:.1f} | {l['two_stage_median']:.1f} | {l['two_stage_p95']:.1f} |",
        f"| Dehaze stage alone (mean) | {l['dehaze_mean']:.1f} | - | - |",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
