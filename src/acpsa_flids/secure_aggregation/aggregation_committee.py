from dataclasses import dataclass, field
import hashlib
from .finite_field import add


@dataclass
class AggregateShare:
    node: int
    clients: frozenset
    value: object
    round_id: int


@dataclass
class AggregationNode:
    node: int
    dimension: int
    round_id: int
    _clients: set = field(default_factory=set)
    _sum: object = None

    def receive(self, client_id, share):
        if client_id in self._clients or share.shape != (self.dimension,):
            raise ValueError("Duplicate client or invalid share dimension")
        self._sum = share.copy() if self._sum is None else add(self._sum, share)
        self._clients.add(client_id)

    def finish(self):
        if self._sum is None:
            raise RuntimeError("No accepted clients")
        return AggregateShare(self.node, frozenset(self._clients), self._sum.copy(), self.round_id)


class AggregationCommittee:
    """In-process transport boundary; nodes retain sums, never individual shares."""

    def __init__(self, nodes, dimension, round_id=0):
        self._nodes = {a: AggregationNode(a, dimension, round_id) for a in range(1, nodes+1)}

    def deliver(self, client_id, shares):
        if set(shares) != set(self._nodes):
            raise ValueError("A contribution must reach the entire accepted committee")
        for node, share in shares.items():
            self._nodes[node].receive(client_id, share)

    def finish(self, unavailable=()):
        return [node.finish() for a, node in self._nodes.items() if a not in unavailable]
