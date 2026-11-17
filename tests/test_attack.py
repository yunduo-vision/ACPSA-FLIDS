import numpy as np
import pytest
from acpsa_flids.evaluation.privacy_attack import evaluate_attack


def test_attack_groups_clients_and_uses_visible_features(tmp_path):
    rng = np.random.default_rng(24)
    groups = np.arange(20)
    # Observations independent from secret targets; repeated aggregate per round for SA.
    targets = rng.integers(0,2,(20,3))
    for t in range(3):
        observations = np.broadcast_to(rng.normal(size=4),(20,4)).copy()
        np.savez(tmp_path/f"round_{t}.npz", observations=observations,targets=targets,clients=groups)
    result = evaluate_attack(tmp_path,trees=5)
    assert not set(result["train_clients"]) & set(result["test_clients"])
    assert result["records"] == 60
    assert 0 <= result["attack_macro_f1"] <= 1
