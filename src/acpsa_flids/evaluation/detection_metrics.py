import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


@torch.no_grad()
def predict_probabilities(model, x, indices=None, batch_size=4096):
    model.eval()
    indices = np.arange(len(x)) if indices is None else np.asarray(indices)
    device = next(model.parameters()).device
    probabilities = []
    for start in range(0, len(indices), batch_size):
        batch = torch.from_numpy(np.asarray(x[indices[start:start+batch_size]], dtype=np.float32).copy()).to(device)
        probabilities.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, model.classes))


def detection_metrics(target, predicted, classes, evaluated=None):
    labels = np.arange(len(classes)) if evaluated is None else np.asarray(evaluated)
    if not len(target):
        return {"samples": 0, "accuracy": None, "macro_precision": None, "macro_recall": None, "macro_f1": None}
    p, r, f, support = precision_recall_fscore_support(target, predicted, labels=labels, zero_division=0)
    return {"samples": len(target), "accuracy": accuracy_score(target, predicted), "macro_precision": p.mean(),
            "macro_recall": r.mean(), "macro_f1": f.mean(),
            "per_class": {classes[int(c)]: {"precision": p[j], "recall": r[j], "f1": f[j], "support": int(support[j]), "fnr": 1-r[j] if support[j] else None} for j, c in enumerate(labels)},
            "confusion_matrix": confusion_matrix(target, predicted, labels=np.arange(len(classes)))}
