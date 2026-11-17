import math
import numpy as np


def epsilon_from_rho(rho, delta):
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    return rho + 2 * math.sqrt(rho * math.log(1 / delta)) if rho else 0.0


def epsilon_for_sigma(sigma, rounds, delta=1e-5):
    if sigma < 0 or rounds < 0:
        raise ValueError("Negative sigma or rounds")
    return epsilon_from_rho(rounds / (2 * sigma**2), delta) if sigma else (math.inf if rounds else 0.0)


def calibrate_sigma(epsilon, rounds, delta=1e-5):
    if epsilon == math.inf:
        return 0.0
    if epsilon <= 0 or rounds <= 0 or not 0 < delta < 1:
        raise ValueError("Invalid privacy calibration")
    L = math.log(1 / delta)
    return math.sqrt(rounds / 2) * (math.sqrt(L + epsilon) + math.sqrt(L)) / epsilon


class RDPAccountant:
    """Continuous-order optimum of the paper's unsampled Gaussian RDP bound."""

    def __init__(self, delta=1e-5, rho=0.0, releases=0):
        if not 0 < delta < 1 or rho < 0:
            raise ValueError("Invalid accountant state")
        self.delta, self.rho, self.releases = delta, rho, releases

    @property
    def epsilon(self):
        return epsilon_from_rho(self.rho, self.delta)

    def projected(self, sigma):
        return epsilon_from_rho(self.rho + 1 / (2 * sigma**2), self.delta) if sigma else math.inf

    def can_release(self, sigma, limit):
        return self.projected(sigma) <= limit + 1e-9

    def step(self, sigma):
        self.rho += 1 / (2 * sigma**2) if sigma else math.inf
        self.releases += 1
        return self.epsilon

    def state_dict(self):
        orders = np.array([1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128, 256])
        return {"delta": self.delta, "rho": self.rho, "releases": self.releases, "epsilon": self.epsilon,
                "orders": orders.tolist(), "rdp": (orders * self.rho).tolist(),
                "scope": "ideal-real-valued Gaussian mechanism; finite precision is not certified"}
