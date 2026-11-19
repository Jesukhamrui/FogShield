"""Drawing and counting helpers shared by the web app."""
import cv2

COLORS = {"person": (0, 255, 255), "vehicle": (0, 255, 0)}  # BGR


def draw_detections(img_bgr, dets):
    out = img_bgr.copy()
    for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
        x1, y1, x2, y2 = map(int, box)
        c = COLORS.get(label, (255, 0, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{label} {score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    return out


def count_labels(dets):
    counts = {"person": 0, "vehicle": 0}
    for l in dets["labels"]:
        counts[l] = counts.get(l, 0) + 1
    return counts


def limit_size(img_bgr, max_side=1280):
    """Shrink very large uploads so CPU processing stays reasonable."""
    h, w = img_bgr.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img_bgr
    f = max_side / s
    return cv2.resize(img_bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def encode_png(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""
