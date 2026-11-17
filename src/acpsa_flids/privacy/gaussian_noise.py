import math
import secrets
import numpy as np


class SecureGaussianRNG:
    """Vectorized Box-Muller from fresh OS entropy, without a reproducible seed."""

    def normal(self, loc=0, scale=1, size=1):
        count = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        pairs = (count+1)//2
        bits = np.frombuffer(secrets.token_bytes(16*pairs), dtype=np.uint64) >> 11
        uniform = (bits.astype(np.float64)+0.5) / 9007199254740992.0
        radius = np.sqrt(-2*np.log(uniform[:pairs]))
        angle = 2*np.pi*uniform[pairs:]
        values = np.concatenate([radius*np.cos(angle), radius*np.sin(angle)])[:count]
        return (loc+scale*values).reshape(size) if isinstance(size, tuple) else loc+scale*values


def private_rng(mode="secure", seed=None):
    if mode not in ("secure", "reproducible"):
        raise ValueError("Unknown noise RNG mode")
    # Each client's seed is private, OS-generated, and never checkpointed in secure mode.
    return SecureGaussianRNG() if mode == "secure" else np.random.default_rng(seed)


def client_noise(dimension, sigma, sensitivity, h_min, rng):
    if sigma < 0 or sensitivity <= 0 or h_min <= 0:
        raise ValueError("Invalid Gaussian mechanism")
    return rng.normal(0, sigma * sensitivity / math.sqrt(h_min), dimension)


def aggregate_variance(contributors, sigma, sensitivity, h_min):
    return contributors / h_min * sigma**2 * sensitivity**2


def check_honest(contributors, collusion_bound, h_min):
    if contributors - collusion_bound < h_min:
        raise RuntimeError(f"Unknown-noise condition failed: {contributors}-{collusion_bound} < {h_min}")
