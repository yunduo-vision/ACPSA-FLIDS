"""The paper's separate controlled Gaussian classification tasks."""
import json
from pathlib import Path
import numpy as np
from acpsa_flids.utils.logger import write_json
from acpsa_flids.data.dataset import PreparedData
from acpsa_flids.evaluation.plots import plot_partition


def generate(config, scenario, task, directory):
    c = config["controlled"]
    classes = c["scenarios"][scenario]
    clients = config["federation"]["clients"]
    seed = config["seed"] + task * 10007 + (0 if scenario == "A" else 10000019)
    rng = np.random.default_rng(seed)
    d, n = c["dimensions"], c["samples_per_client"]
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if (directory/"metadata.json").exists():
        old = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
        if old["generation_config"] != {"controlled": c, "clients": clients, "alpha": config["federation"]["alpha"], "seed": seed}:
            raise ValueError("Controlled task cache differs; use a new task directory")
        return PreparedData(directory)
    means = rng.normal(0, c["mean_scale"], (classes, d))
    probabilities = rng.uniform(*c["visibility_range"], classes-1)
    visibility = np.ones((clients, classes), dtype=bool)
    visibility[:, 1:] = rng.random((clients, classes-1)) < probabilities
    for j in range(1, classes):
        if not visibility[:, j].any():
            visibility[rng.integers(clients), j] = True
    all_x, all_y, parts, offset = [], [], [], 0
    for i in range(clients):
        visible = np.flatnonzero(visibility[i])
        proportions = rng.dirichlet(np.full(len(visible), config["federation"]["alpha"]))
        if n < len(visible):
            raise ValueError("Too few samples to realize selected visible classes")
        counts = 1+rng.multinomial(n-len(visible), proportions)
        train, validation = [], []
        for label, count in zip(visible, counts):
            all_x.append(rng.normal(means[label], c["feature_std"], (count, d)).astype(np.float32))
            all_y.append(np.full(count, label, dtype=np.int64))
            indices = rng.permutation(np.arange(offset, offset+count))
            nval = min(count-1, max(1, round(count*config["data"]["validation_fraction"]))) if count > 1 else 0
            train.extend(indices[nval:]); validation.extend(indices[:nval]); offset += count
        parts.append({"train": np.array(train, dtype=np.int64), "validation": np.array(validation, dtype=np.int64)})
    test_indices = []
    for label in range(classes):
        count = c["test_per_class"]
        all_x.append(rng.normal(means[label], c["feature_std"], (count, d)).astype(np.float32))
        all_y.append(np.full(count, label, dtype=np.int64))
        test_indices.extend(range(offset, offset+count)); offset += count
    x, y = np.concatenate(all_x), np.concatenate(all_y)
    all_train = np.concatenate([p["train"] for p in parts])
    # Raw Gaussian features are used directly: these controlled tasks are not network-flow preprocessing.
    np.save(directory/"features.npy", x); np.save(directory/"labels.npy", y)
    regular = np.concatenate([p["validation"] for p in parts])
    if not len(regular):
        raise ValueError("Controlled configuration produces no validation records")
    np.savez_compressed(directory/"splits.npz", train=all_train, full=np.array(test_indices), regular=regular)
    np.savez_compressed(directory/"clients.npz", **{f"{name}_{i}": idx for i, part in enumerate(parts) for name, idx in part.items()}, visibility=visibility)
    names = ["Benign"]+[f"C{i}" for i in range(1, classes)]
    write_json(directory/"metadata.json", {"dataset": f"controlled_{scenario}", "task": task, "seed": seed,
        "clients": clients, "classes": names, "preprocessor": None, "label_mapping": {},
        "visibility_probabilities": probabilities, "means": means,
        "generation_config": {"controlled": c, "clients": clients, "alpha": config["federation"]["alpha"], "seed": seed},
        "minimum_allocation": "one record per Bernoulli-selected class, then multinomial Dirichlet allocation"})
    plot_partition(y, parts, names, directory)
    return PreparedData(directory)
