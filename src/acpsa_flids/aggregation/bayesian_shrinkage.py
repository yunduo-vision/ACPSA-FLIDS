import numpy as np


def shrink(counts, head_sum, contributors, noise_variance, prior_variance, min_support=1.0):
    n = np.clip(np.asarray(counts, dtype=np.float64), 0, contributors)
    s2 = np.broadcast_to(np.asarray(prior_variance, dtype=np.float64), n.shape)
    if np.any(s2 <= 0) or noise_variance < 0:
        raise ValueError("Invalid Gaussian variance")
    numerator = n**2 * s2
    denominator = numerator + noise_variance
    weight = np.divide(numerator, denominator, out=np.zeros_like(n), where=denominator > 0)
    active = n > min_support
    direct = np.divide(head_sum, n[:, None], out=np.zeros_like(head_sum, dtype=float), where=active[:, None])
    update = weight[:, None] * direct
    return update, n, weight, direct


class PriorVariance:
    def __init__(self, classes, config):
        self.config = config
        self.values = np.broadcast_to(np.asarray(config["prior_variance"], dtype=float), (classes,)).copy()
        if config["prior_mode"] == "public":
            from ..utils.config import resolve
            self.values = np.asarray(np.load(resolve(config["public_prior_file"])), dtype=float)
        if self.values.shape != (classes,) or np.any(self.values <= 0):
            raise ValueError("Invalid public/initial prior variances")
        if config["prior_mode"] not in ("fixed", "history", "public"):
            raise ValueError("Unknown prior mode")

    def observe(self, direct, support, noise_variance):
        # Called AFTER current-round shrinkage; uses only already released statistics.
        if self.config["prior_mode"] != "history":
            return
        active = support > self.config["min_support"]
        estimate = np.mean(direct**2, axis=1) - noise_variance / np.maximum(support, 1)**2
        estimate = np.clip(estimate, self.config["prior_floor"], self.config["prior_ceiling"])
        decay = self.config["prior_decay"]
        self.values[active] = decay * self.values[active] + (1-decay) * estimate[active]
