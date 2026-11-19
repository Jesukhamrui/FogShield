"""Metrics: PSNR, SSIM, IoU and VOC-style mAP@0.5 for detection."""
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), both in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def map50(predictions: list, ground_truths: list, iou_thr: float = 0.5):
    """predictions[i] = {'boxes','scores','labels'}; ground_truths[i] = {'boxes','labels'}.

    Returns (mAP over classes present in ground truth, per-class AP dict).
    """
    classes = sorted({l for gt in ground_truths for l in gt["labels"]})
    per_class = {}
    for c in classes:
        gt_boxes = {}
        npos = 0
        for i, gt in enumerate(ground_truths):
            sel = np.array([l == c for l in gt["labels"]], dtype=bool)
            gt_boxes[i] = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)[sel]
            npos += int(sel.sum())
        if npos == 0:
            continue

        dets = []
        for i, p in enumerate(predictions):
            for box, score, label in zip(p["boxes"], p["scores"], p["labels"]):
                if label == c:
                    dets.append((float(score), i, box))
        if not dets:
            per_class[c] = 0.0
            continue
        dets.sort(key=lambda d: -d[0])

        matched = {i: np.zeros(len(g), dtype=bool) for i, g in gt_boxes.items()}
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for k, (_, i, box) in enumerate(dets):
            g = gt_boxes[i]
            if len(g) == 0:
                fp[k] = 1
                continue
            ious = iou_matrix(box[None], g)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched[i][j]:
                tp[k] = 1
                matched[i][j] = True
            else:
                fp[k] = 1

        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / npos
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        per_class[c] = _voc_ap(rec, prec)

    mAP = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mAP, per_class
