import numpy as np


def clip_block(vector, bound):
    if bound <= 0:
        raise ValueError("Clipping bound must be positive")
    vector = np.asarray(vector, dtype=np.float64)
    before = float(np.linalg.norm(vector))
    if not np.isfinite(before):
        raise ValueError("Non-finite update")
    ratio = min(1.0, bound / before) if before else 1.0
    clipped = vector * ratio
    return clipped, {"before": before, "after": float(np.linalg.norm(clipped)), "ratio": ratio}
