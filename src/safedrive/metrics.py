"""Model metrics with explicit safe-frame false-positive denominator."""
import numpy as np


def classification_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if (y.ndim != 1 or len(y) == 0 or p.shape != (len(y), 10) or
        not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or
        not np.allclose(p.sum(1), 1, atol=1e-5) or ((y < 0) | (y > 9)).any()):
        raise ValueError("Invalid labels or probabilities")
    pred = p.argmax(1)
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = cm.diagonal()
    support = cm.sum(1)
    recalls = np.divide(tp, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2*tp, cm.sum(0)+support, out=np.zeros(10), where=(cm.sum(0)+support) != 0)
    safe = y == 0
    flagged = (1-p[:, 0]) >= 0.5
    return {"images": len(y), "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            "accuracy": float((pred == y).mean()), "macro_f1": float(f1.mean()),
            "per_class_recall": recalls.tolist(), "per_class_support": support.tolist(),
            "confusion_matrix": cm.tolist(), "frame_threshold": 0.5,
            "safe_frame_false_positive_rate": float(flagged[safe].mean()) if safe.any() else None,
            "non_safe_frame_recall": float(flagged[~safe].mean()) if (~safe).any() else None}
