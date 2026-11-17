import re
import numpy as np
import pandas as pd


def normalize_column(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def numeric_frame(frame, label_column, drop_patterns):
    columns = [normalize_column(c) for c in frame.columns]
    frame = frame.copy()
    frame.columns = columns
    frame = frame.loc[:, ~frame.columns.duplicated()]
    label = normalize_column(label_column)
    if label not in frame:
        raise ValueError(f"Required label column {label!r} missing")
    labels = frame[label].astype(str)
    selected = [c for c in frame if c != label and not any(re.search(p, c) for p in drop_patterns)]
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric, labels


class FeaturePreprocessor:
    """Train-only medians, nonnegative log1p selection and population Z scores."""

    def fit(self, x, columns, log_patterns):
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)
        self.columns = list(columns)
        self.keep = np.ones(x.shape[1], dtype=bool)
        # Drop exact duplicated feature columns using training data alone.
        seen = {}
        for j in range(x.shape[1]):
            key = np.nan_to_num(x[:, j], nan=-1.23456789012345e300).tobytes()
            if key in seen:
                self.keep[j] = False
            else:
                seen[key] = j
        x = x[:, self.keep].copy()
        self.feature_names = [c for c, keep in zip(columns, self.keep) if keep]
        self.medians = np.array([np.nanmedian(x[:, j]) if np.isfinite(x[:, j]).any() else 0.0 for j in range(x.shape[1])])
        x = np.where(np.isfinite(x), x, self.medians)
        self.log_mask = np.array([np.min(x[:, j]) >= 0 and any(re.search(p, name) for p in log_patterns) for j, name in enumerate(self.feature_names)])
        x[:, self.log_mask] = np.log1p(x[:, self.log_mask])
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)[:, self.keep].copy()
        x = np.where(np.isfinite(x), x, self.medians)
        # New negative readings in a train-nonnegative feature are treated as invalid.
        log_x = x[:, self.log_mask]
        log_x = np.where(log_x >= 0, log_x, self.medians[self.log_mask])
        x[:, self.log_mask] = np.log1p(log_x)
        return ((x - self.mean) / self.std).astype(np.float32)

    def transform_frame(self, frame):
        frame = frame.copy()
        frame.columns = [normalize_column(c) for c in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()]
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing inference features: {sorted(missing)}")
        x = frame[self.columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        return self.transform(x)

    def state_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    @classmethod
    def from_state(cls, state):
        instance = cls()
        for k, value in state.items():
            setattr(instance, k, np.asarray(value) if k not in ("columns", "feature_names") else value)
        return instance
