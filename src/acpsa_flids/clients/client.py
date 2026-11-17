import copy
import time
import numpy as np
import torch
from .local_trainer import local_train
from ..models.model_utils import blocks
from ..semantics.synthetic_probing import probe
from ..privacy.gaussian_noise import client_noise, private_rng
from ..evaluation.detection_metrics import predict_probabilities


def observation_features(vector, layout):
    """Adversary-visible vector features only; no local labels or thresholds."""
    feature, indicator, head = layout.unpack(vector)
    return np.concatenate([indicator, np.linalg.norm(head, axis=1), head.mean(axis=1),
                           [np.linalg.norm(feature), feature.mean(), feature.std()]])


class FederatedClient:
    def __init__(self, identity, data, config, layout):
        self.identity, self.data, self.config, self.layout = identity, data, config, layout
        self.part = data.clients[identity]
        self.observed = np.isin(np.arange(layout.classes), np.unique(data.y[self.part["train"]])).astype(float)
        self._noise_rng = private_rng(config["privacy"]["noise_rng"], config["seed"] + 1000003 * (identity+1))

    def execute(self, global_model, round_id, sigma, protocol, committee, contributors, diagnostic_sink=None, attack_sink=None):
        config, layout = self.config, self.layout
        seed = config["seed"] + round_id * 100003 + self.identity * 101
        local = copy.deepcopy(global_model)
        before_feature, before_head = blocks(global_model)
        loss = local_train(local, self.data.x, self.data.y, self.part["train"], config["local"], seed)
        indicator, confidence, threshold = probe(local, self.observed, config["probing"]["count"], config["probing"]["thresholds"], torch.Generator().manual_seed(seed+1))
        after_feature, after_head = blocks(local)
        start = time.perf_counter()
        vector, clipping = layout.build(after_feature-before_feature, after_head-before_head, indicator)
        method = config["aggregation"]["method"]
        if method in ("global", "confidence", "precision"):
            if config["variant"] != "fid_spa":
                raise ValueError("Weighted/FedAvg comparisons require nonprivate fid_spa variant; no unproved DP claims")
            weights = np.ones(layout.classes) if method == "global" else confidence * indicator
            if method == "precision":
                val = self.part["validation"]
                precision = np.zeros(layout.classes)
                if len(val):
                    predicted = predict_probabilities(local, self.data.x, val, config["evaluation"]["batch_size"]).argmax(axis=1)
                    target = self.data.y[val]
                    for c in range(layout.classes):
                        precision[c] = np.sum((predicted == c) & (target == c)) / max(np.sum(predicted == c), 1)
                weights = precision * indicator
            feature, _, head = layout.unpack(vector)
            if method == "global":
                # Rebuild with all class blocks instead of reusing masked blocks.
                vector, _ = layout.build(after_feature-before_feature, after_head-before_head, np.ones(layout.classes))
                feature, _, head = layout.unpack(vector)
            else:
                # Rebuild clipped blocks once; weight multiplication is <=1.
                _, _, head = layout.unpack(layout.build(after_feature-before_feature, after_head-before_head, np.ones(layout.classes))[0])
            vector = layout.pack(feature, weights, weights[:, None] * head)
        variant = config["variant"]
        noise_sensitivity = layout.sensitivity
        noise = np.zeros(layout.dimension)
        if variant == "acpsa":
            noise = client_noise(layout.dimension, sigma, layout.sensitivity, config["privacy"]["h_min"], self._noise_rng)
        elif variant in ("indicator_dp", "update_dp"):
            mask = np.zeros(layout.dimension, dtype=bool)
            mask[layout.feature_dim:layout.feature_dim+layout.classes] = True
            if variant == "update_dp":
                mask = ~mask
                noise_sensitivity = 2 * np.sqrt(layout.feature_bound**2 + np.sum(layout.class_bounds**2))
            else:
                noise_sensitivity = 2 * np.sqrt(layout.classes)
            noise[mask] = client_noise(int(mask.sum()), sigma, noise_sensitivity, config["privacy"]["h_min"], self._noise_rng)
        elif variant not in ("fid_spa", "full_sa"):
            raise ValueError("Unknown protection variant")
        protected = vector + noise
        noise_time = time.perf_counter()-start
        start = time.perf_counter()
        if protocol is not None:
            shares = protocol.share(protected, contributors)
            sharing_time = time.perf_counter()-start
            start = time.perf_counter()
            committee.deliver(self.identity, shares)
            delivery_time = time.perf_counter()-start
            upload_bytes = sum(v.nbytes for v in shares.values())
        else:
            # This branch is an explicitly plaintext ablation, never ACPSA/Full-SA.
            committee.receive(self.identity, protected)
            sharing_time, delivery_time, upload_bytes = 0.0, 0.0, protected.nbytes
        if attack_sink is not None:
            attack_sink(self.identity, self.observed, None if protocol is not None else observation_features(protected, layout))
        if diagnostic_sink is not None:
            diagnostic_sink(self.identity, {"round": round_id, "loss": loss, "clipping": clipping,
                             "protection_seconds": noise_time, "sharing_seconds": sharing_time,
                             "node_sum_seconds": delivery_time, "upload_bytes": upload_bytes})
        del local
        return {"protection_seconds": noise_time, "sharing_seconds": sharing_time,
                "node_sum_seconds": delivery_time, "upload_bytes": upload_bytes}
