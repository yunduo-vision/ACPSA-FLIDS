from dataclasses import dataclass
import numpy as np
from ..privacy.clipping import clip_block


@dataclass
class ContributionLayout:
    feature_dim: int
    classes: int
    head_dim: int
    feature_bound: float
    class_bounds: object

    def __post_init__(self):
        self.class_bounds = np.broadcast_to(np.asarray(self.class_bounds, dtype=float), (self.classes,)).copy()
        if self.feature_bound <= 0 or np.any(self.class_bounds <= 0):
            raise ValueError("All block bounds must be positive")

    @property
    def dimension(self):
        return self.feature_dim + self.classes + self.classes * self.head_dim

    @property
    def radius(self):
        return np.sqrt(self.feature_bound**2 + self.classes + np.sum(self.class_bounds**2))

    @property
    def sensitivity(self):
        return 2 * self.radius

    def pack(self, feature, indicator, head):
        v = np.concatenate([np.asarray(feature).ravel(), np.asarray(indicator).ravel(), np.asarray(head).ravel()])
        if v.size != self.dimension or not np.isfinite(v).all():
            raise ValueError("Invalid contribution layout")
        return v

    def unpack(self, vector):
        v = np.asarray(vector)
        if v.shape != (self.dimension,):
            raise ValueError("Joint vector dimension mismatch")
        j = self.feature_dim
        return v[:j], v[j:j+self.classes], v[j+self.classes:].reshape(self.classes, self.head_dim)

    def build(self, delta_feature, delta_head, indicator):
        b = np.asarray(indicator)
        if b.shape != (self.classes,) or not np.isin(b, [0, 1]).all():
            raise ValueError("Knowledge indicators must be binary")
        feature, fs = clip_block(delta_feature, self.feature_bound)
        heads, hs = [], []
        for c in range(self.classes):
            head, stats = clip_block(delta_head[c], self.class_bounds[c])
            heads.append(head * b[c])
            hs.append(stats)
        v = self.pack(feature, b, np.stack(heads))
        if np.linalg.norm(v) > self.radius + 1e-9:
            raise AssertionError("Joint contribution exceeds sensitivity bound")
        return v, {"feature": fs, "classes": hs}
