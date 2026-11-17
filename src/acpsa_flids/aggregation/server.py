import numpy as np
from ..models.model_utils import blocks, set_blocks
from .bayesian_shrinkage import shrink


def update_model(model, layout, aggregate, contributors, variance, prior, config):
    feature_sum, count_sum, head_sum = layout.unpack(aggregate)
    update, support, weight, direct = shrink(count_sum, head_sum, contributors, variance, prior.values, config["min_support"])
    method = config["method"]
    if method in ("classwise", "confidence", "precision", "global"):
        # Weighted baselines encode their weights in count_sum and head_sum locally.
        update, weight = direct, (support > config["min_support"]).astype(float)
    elif method != "bayesian":
        raise ValueError("Unknown aggregation method")
    feature, head = blocks(model)
    feature_delta = config["feature_lr"] * feature_sum / contributors
    class_lr = np.broadcast_to(np.asarray(config["class_lr"]), (layout.classes,))
    head_delta = class_lr[:, None] * update
    if not np.isfinite(feature_delta).all() or not np.isfinite(head_delta).all():
        raise FloatingPointError("Non-finite global update")
    set_blocks(model, feature + feature_delta, head + head_delta)
    prior.observe(direct, support, variance)
    return {"support": support, "shrinkage": weight, "prior_variance_next": prior.values.copy(),
            "feature_update_norm": np.linalg.norm(feature_delta), "class_update_norms": np.linalg.norm(head_delta, axis=1)}
