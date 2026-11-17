import numpy as np


class PlaintextAccumulator:
    """Explicit unprotected ablation transport; never selected by the ACPSA variant."""
    def __init__(self, dimension):
        self.sum = np.zeros(dimension)
        self.clients = set()

    def receive(self, client_id, vector):
        if client_id in self.clients or vector.shape != self.sum.shape:
            raise ValueError("Duplicate client or invalid plaintext vector")
        self.sum += vector
        self.clients.add(client_id)
