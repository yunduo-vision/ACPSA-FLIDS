import time
from .finite_field import FixedPointField
from .shamir_secret_sharing import split, reconstruct


class SecureAggregation:
    def __init__(self, nodes=5, threshold=3, scale=10000, prime=2147483647, enabled=True):
        if not 2 <= threshold <= nodes <= 32:
            raise ValueError("Invalid committee parameters")
        self.nodes, self.threshold, self.field = nodes, threshold, FixedPointField(scale, prime)

    def share(self, vector, max_contributors):
        return split(self.field.encode(vector, max_contributors), self.nodes, self.threshold)

    def recover(self, aggregate_shares, accepted_clients, round_id=0):
        if len(aggregate_shares) < self.threshold:
            raise RuntimeError("Committee availability below reconstruction threshold")
        seen = set()
        for s in aggregate_shares:
            if s.node in seen or not 1 <= s.node <= self.nodes or s.clients != frozenset(accepted_clients) or s.round_id != round_id:
                raise RuntimeError("Committee accepted-client-set or round agreement failed")
            seen.add(s.node)
        return self.field.decode(reconstruct({s.node: s.value for s in aggregate_shares}, self.threshold))
