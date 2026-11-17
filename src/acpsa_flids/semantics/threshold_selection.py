import numpy as np


def select_threshold(confidence, reference, candidates):
    q, z = np.asarray(confidence), np.asarray(reference, dtype=bool)
    if q.shape != z.shape or not candidates:
        raise ValueError("Invalid threshold inputs")
    best = None
    for tau in sorted(candidates):
        predicted = q >= tau
        tp, fp, fn = (int(np.sum(predicted & z)), int(np.sum(predicted & ~z)), int(np.sum(~predicted & z)))
        score = 2 * tp / max(2 * tp + fp + fn, 1)
        key = (score, -fp, tau)  # last deterministic tie: higher threshold
        if best is None or key > best[0]:
            best = (key, tau, predicted.astype(np.float64))
    return best[1], best[2]
