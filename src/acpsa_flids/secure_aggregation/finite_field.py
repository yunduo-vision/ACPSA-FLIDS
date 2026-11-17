"""Vectorized arithmetic in F_(2^31-1), with safe signed int64 products."""
import numpy as np

PRIME = 2147483647


class FixedPointField:
    def __init__(self, scale=10000, prime=PRIME):
        if prime != PRIME or scale <= 0:
            raise ValueError("Only the verified Mersenne prime 2^31-1 is supported")
        self.prime, self.scale = prime, scale

    def encode(self, vector, max_contributors):
        x = np.asarray(vector, dtype=np.float64)
        if max_contributors <= 0 or not np.isfinite(x).all():
            raise ValueError("Invalid finite-field input")
        limit = (self.prime // 2) // max_contributors
        if np.max(np.abs(x), initial=0) * self.scale + 0.5 > limit:
            raise OverflowError("Signed aggregate may wrap; reduce scale or revise public encoding bound")
        integers = np.rint(x * self.scale).astype(np.int64)
        return integers % self.prime

    def decode(self, vector):
        v = np.asarray(vector, dtype=np.int64)
        signed = np.where(v > self.prime // 2, v - self.prime, v)
        return signed.astype(np.float64) / self.scale


def add(a, b, prime=PRIME):
    return (a + b) % prime


def multiply(a, b, prime=PRIME):
    # (p-1)^2 < 2^62; addition is performed separately after modular reduction.
    return (np.asarray(a, dtype=np.int64) * b) % prime
