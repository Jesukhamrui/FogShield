import numpy as np

from src.metrics import iou_matrix, map50, psnr


def test_iou_identical_boxes_is_one():
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert np.isclose(iou_matrix(b, b)[0, 0], 1.0)


def test_iou_disjoint_boxes_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_map50_perfect_predictions():
    gts = [{"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "labels": ["vehicle", "person"]}]
    preds = [{"boxes": np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
              "scores": np.array([0.9, 0.8]), "labels": ["vehicle", "person"]}]
    m, per = map50(preds, gts)
    assert np.isclose(m, 1.0)
    assert set(per) == {"vehicle", "person"}


def test_map50_no_predictions_is_zero():
    gts = [{"boxes": [[0, 0, 10, 10]], "labels": ["vehicle"]}]
    preds = [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": []}]
    m, _ = map50(preds, gts)
    assert m == 0.0


def test_psnr_identical_is_inf():
    a = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert psnr(a, a) == float("inf")
