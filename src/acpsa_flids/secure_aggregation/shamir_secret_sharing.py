import secrets
import numpy as np
from .finite_field import PRIME, add, multiply


def random_field(size):
    """Uniform CSPRNG field elements using vectorized rejection sampling."""
    result = np.frombuffer(secrets.token_bytes(4 * size), dtype=np.uint32).astype(np.int64) & PRIME
    rejected = result == PRIME
    while np.any(rejected):
        result[rejected] = np.frombuffer(secrets.token_bytes(4 * int(rejected.sum())), dtype=np.uint32).astype(np.int64) & PRIME
        rejected = result == PRIME
    return result


def split(secret, nodes=5, threshold=3):
    if not 2 <= threshold <= nodes < PRIME:
        raise ValueError("Invalid Shamir threshold")
    secret = np.asarray(secret, dtype=np.int64)
    if secret.ndim != 1 or np.any(secret < 0) or np.any(secret >= PRIME):
        raise ValueError("Secret must be a vector of field elements")
    coefficients = [secret] + [random_field(secret.size) for _ in range(threshold - 1)]
    shares = {}
    for x in range(1, nodes + 1):
        y = coefficients[-1].copy()
        for a in reversed(coefficients[:-1]):
            y = add(multiply(y, x), a)
        shares[x] = y
    return shares


def reconstruct(shares, threshold=3):
    if len(shares) < threshold:
        raise RuntimeError("Insufficient shares for reconstruction")
    xs = list(shares)[:threshold]
    if len(set(xs)) != threshold or any(x <= 0 or x >= PRIME for x in xs):
        raise ValueError("Invalid share coordinates")
    shape = np.asarray(shares[xs[0]]).shape
    result = np.zeros(shape, dtype=np.int64)
    for x in xs:
        y = np.asarray(shares[x], dtype=np.int64)
        if y.shape != shape or np.any(y < 0) or np.any(y >= PRIME):
            raise ValueError("Invalid share vector")
        numerator, denominator = 1, 1
        for other in xs:
            if other != x:
                numerator = numerator * (-other) % PRIME
                denominator = denominator * (x - other) % PRIME
        weight = numerator * pow(denominator, -1, PRIME) % PRIME
        result = add(result, multiply(y, weight))
    # Additional completing nodes must agree with the same aggregate polynomial.
    for extra in list(shares)[threshold:]:
        predicted = np.zeros(shape, dtype=np.int64)
        for x in xs:
            coefficient = 1
            for other in xs:
                if x != other:
                    coefficient = coefficient * (extra - other) * pow(x - other, -1, PRIME) % PRIME
            predicted = add(predicted, multiply(shares[x], coefficient))
        if not np.array_equal(predicted, shares[extra]):
            raise RuntimeError("Inconsistent aggregate shares")
    return result
