import numpy as np
from .detection_metrics import detection_metrics


def unseen_metrics(target, predicted, data):
    global_classes = set(np.unique(data.y[data.splits["train"]])) - {0}
    rows = []
    for i, part in enumerate(data.clients):
        missing = sorted(global_classes - set(np.unique(data.y[part["train"]])))
        mask = np.isin(target, missing)
        row = detection_metrics(target[mask], predicted[mask], data.classes, missing)
        rows.append({"client": i, "unseen_classes": [data.classes[c] for c in missing], **row})
    valid = [r for r in rows if r["samples"]]
    summary = {k: float(np.mean([r[k] for r in valid])) if valid else None for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    return {"client_mean": summary, "clients_evaluated": len(valid), "per_client": rows,
            "definition": "global-model predictions on federation-seen but client-training-absent attack classes"}
