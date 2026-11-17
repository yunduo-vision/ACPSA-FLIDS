import numpy as np


def stratified_splits(y, config, seed):
    rng = np.random.default_rng(seed)
    train, regular, full, unused = [], [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        if len(idx) < 6:
            raise ValueError(f"Class {c} has fewer than 6 records; cannot preserve all splits")
        nf = max(1, int(round(len(idx) * config["full_test_fraction"])))
        rest = idx[nf:]
        nr = max(1, int(round(len(rest) * config["regular_test_fraction"])))
        candidate = rest[nr:]
        nt = max(1, int(round(len(candidate) * config["train_fraction"])))
        full.extend(idx[:nf]); regular.extend(rest[:nr]); train.extend(candidate[:nt]); unused.extend(candidate[nt:])
    return {"train": np.array(train, dtype=np.int64), "regular": np.array(regular, dtype=np.int64),
            "full": np.array(full, dtype=np.int64), "unused": np.array(unused, dtype=np.int64)}


def partition_clients(y, train_indices, clients, classes, benign, alpha, max_attack_classes, validation_fraction, seed):
    if alpha <= 0 or max_attack_classes < 1 or clients * max_attack_classes < classes-1:
        raise ValueError("Infeasible visibility/Dirichlet settings")
    rng = np.random.default_rng(seed)
    allowed = np.zeros((clients, classes), dtype=bool)
    allowed[:, benign] = True
    attacks = [c for c in range(classes) if c != benign]
    # Cover every global class before filling remaining public visibility slots.
    load = np.zeros(clients, dtype=int)
    for c in rng.permutation(attacks):
        eligible = np.flatnonzero(load < max_attack_classes)
        i = int(rng.choice(eligible))
        allowed[i, c], load[i] = True, load[i]+1
    for i in range(clients):
        remaining = [c for c in attacks if not allowed[i, c]]
        slots = min(max_attack_classes-load[i], len(remaining))
        if slots:
            allowed[i, rng.choice(remaining, size=slots, replace=False)] = True
    # Client-local Dirichlet propensities, then column normalization conserves global counts.
    propensities = np.zeros((clients, classes))
    for i in range(clients):
        visible = np.flatnonzero(allowed[i])
        propensities[i, visible] = rng.dirichlet(np.full(len(visible), alpha))
    buckets = [[] for _ in range(clients)]
    for c in range(classes):
        indices = rng.permutation(train_indices[y[train_indices] == c])
        eligible = np.flatnonzero(allowed[:, c])
        if c == benign:
            if len(indices) < clients:
                raise ValueError("Not enough Benign samples to give one to each client")
            for i, j in enumerate(indices[:clients]):
                buckets[i].append(int(j))
            indices = indices[clients:]
        weights = np.maximum(propensities[eligible, c], np.finfo(float).tiny)
        sizes = rng.multinomial(len(indices), weights / weights.sum())
        offset = 0
        for i, count in zip(eligible, sizes):
            buckets[i].extend(indices[offset:offset+count].tolist())
            offset += count
    assignments, visibility = [], np.zeros((clients, classes), dtype=int)
    for i, bucket in enumerate(buckets):
        local_train, validation = [], []
        # Reserve validation while retaining at least one local optimization sample per class.
        for c in np.unique(y[bucket]):
            idx = rng.permutation(np.array(bucket)[y[bucket] == c])
            n = min(len(idx)-1, max(1, int(round(len(idx)*validation_fraction)))) if len(idx) > 1 else 0
            validation.extend(idx[:n]); local_train.extend(idx[n:])
        train = np.array(local_train, dtype=np.int64)
        assignments.append({"train": train, "validation": np.array(validation, dtype=np.int64)})
        visibility[i, np.unique(y[train])] = 1
    all_indices = np.concatenate([np.concatenate([a["train"], a["validation"]]) for a in assignments])
    if not np.array_equal(np.sort(all_indices), np.sort(train_indices)) or len(np.unique(all_indices)) != len(all_indices):
        raise AssertionError("Client partition lost or duplicated samples")
    if not visibility[:, benign].all() or not visibility.any(axis=0).all() or np.any(visibility[:, attacks].sum(axis=1) > max_attack_classes):
        raise AssertionError("Class visibility constraints violated")
    return assignments, visibility, allowed
