from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from ..utils.logger import write_json


def evaluate_attack(audit_dir, seed=42, test_fraction=0.3, trees=200, max_depth=8, min_samples_leaf=2):
    """Auxiliary-client supervised attack; all rounds of a client stay in one split."""
    audit_dir = Path(audit_dir)
    files = sorted(audit_dir.glob("round_*.npz"))
    if not files:
        raise FileNotFoundError("No opted-in private research audit; enable evaluation.private_audit")
    records = [np.load(p) for p in files]
    x = np.concatenate([r["observations"] for r in records])
    target = np.concatenate([r["targets"] for r in records]).astype(int)
    groups = np.concatenate([r["clients"] for r in records])
    if len(np.unique(groups)) < 4:
        raise ValueError("Class-inference evaluation needs at least four distinct auxiliary/target clients")
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(x, target, groups))
    estimator = RandomForestClassifier(n_estimators=trees, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=2)
    estimator.fit(x[train], target[train])
    predicted = estimator.predict(x[test])
    p, r, f, support = precision_recall_fscore_support(target[test], predicted, average=None, zero_division=0)
    result = {"attack_accuracy": accuracy_score(target[test].ravel(), predicted.ravel()),
              "attack_exact_match_accuracy": accuracy_score(target[test], predicted),
              "attack_macro_precision": p.mean(), "attack_macro_recall": r.mean(), "attack_macro_f1": f.mean(),
              "per_class_f1": f, "positive_support": support, "train_clients": np.unique(groups[train]),
              "test_clients": np.unique(groups[test]), "records": len(x),
              "protocol": "RF auxiliary-client multilabel attack; true coverage labels are supervision/held-out targets only; SA exposes aggregate summaries only"}
    write_json(audit_dir.parent / "privacy_attack.json", result)
    return result
