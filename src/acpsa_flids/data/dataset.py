from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
from ..utils.config import resolve
from ..utils.logger import write_json
from .preprocessing import numeric_frame, FeaturePreprocessor
from .label_mapping import map_labels
from .partition import stratified_splits, partition_clients


class PreparedData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        self.x = np.load(self.directory / "features.npy", mmap_mode="r")
        self.y = np.load(self.directory / "labels.npy", mmap_mode="r")
        self.splits = dict(np.load(self.directory / "splits.npz"))
        parts = np.load(self.directory / "clients.npz")
        self.clients = [{"train": parts[f"train_{i}"], "validation": parts[f"validation_{i}"]} for i in range(self.metadata["clients"])]
        self.classes = self.metadata["classes"]
        self.preprocessor_state = self.metadata["preprocessor"]


def prepare(config):
    raw, out = resolve(config["data"]["raw_dir"]), resolve(config["data"]["processed_dir"])
    files = sorted(p for p in raw.rglob("*.csv") if p.name not in config["data"].get("exclude_files", []))
    if not files:
        raise FileNotFoundError(f"No real CSV files in {raw}; run download_data.py or place official CSVs there")
    out.mkdir(parents=True, exist_ok=True)
    config_key = {"dataset": config["dataset"], "data": config["data"], "federation": config["federation"], "seed": config["seed"]}
    fingerprint = hashlib.sha256(json.dumps(config_key, sort_keys=True).encode()).hexdigest()
    if (out / "metadata.json").exists():
        previous = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        sources = [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]
        if previous["fingerprint"] != fingerprint or previous["sources"] != sources:
            raise ValueError("Cache differs from config or raw files; select a NEW processed_dir")
        return PreparedData(out)
    chunks, all_labels, originals, columns = [], [], {}, None
    for file in files:
        for frame in pd.read_csv(file, chunksize=config["data"]["chunksize"], encoding=config["data"]["encoding"], low_memory=False):
            numeric, raw_labels = numeric_frame(frame, config["data"]["label_column"], config["data"]["drop_patterns"])
            if columns is None:
                columns = numeric.columns.tolist()
            if set(numeric.columns) != set(columns):
                raise ValueError(f"Feature schema differs in {file}")
            labels = map_labels(raw_labels, config["dataset"])
            originals.update(dict(zip(raw_labels.tolist(), labels)))
            chunk = out / f"raw_chunk_{len(chunks):05d}.npy"
            with np.errstate(over="ignore", invalid="ignore"):
                arr = numeric[columns].to_numpy(dtype=np.float32)
            arr[~np.isfinite(arr)] = np.nan
            if shutil.disk_usage(out).free < arr.nbytes * 2 + 100_000_000:
                raise OSError("Not enough cache space; set processed_dir to a larger drive")
            np.save(chunk, arr)
            chunks.append((chunk, len(arr)))
            all_labels.extend(labels)
    classes = ["Benign"] + sorted(set(all_labels) - {"Benign"})
    if "Benign" not in all_labels:
        raise ValueError("Dataset must contain Benign")
    encoding = {name: c for c, name in enumerate(classes)}
    y = np.array([encoding[v] for v in all_labels], dtype=np.int64)
    splits = stratified_splits(y, config["data"], config["seed"])
    parts, visibility, allowed = partition_clients(y, splits["train"], config["federation"]["clients"], len(classes), 0,
        config["federation"]["alpha"], config["federation"]["max_attack_classes"], config["data"]["validation_fraction"], config["seed"])
    fit_indices = np.sort(np.concatenate([p["train"] for p in parts]))
    selected = []
    offset = 0
    for path, length in chunks:
        local = fit_indices[(fit_indices >= offset) & (fit_indices < offset+length)] - offset
        selected.append(np.load(path, mmap_mode="r")[local])
        offset += length
    processor = FeaturePreprocessor().fit(np.concatenate(selected), columns, config["data"]["log_patterns"])
    del selected
    bytes_needed = len(y) * len(processor.feature_names) * 4
    if shutil.disk_usage(out).free < bytes_needed + 100_000_000:
        raise OSError("Not enough space for transformed data; choose a larger processed_dir")
    x = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float32, shape=(len(y), len(processor.feature_names)))
    offset = 0
    for path, length in chunks:
        x[offset:offset+length] = processor.transform(np.load(path))
        offset += length
    x.flush(); del x
    np.save(out / "labels.npy", y)
    np.savez_compressed(out / "splits.npz", **splits)
    np.savez_compressed(out / "clients.npz", **{f"{name}_{i}": indices for i, part in enumerate(parts) for name, indices in part.items()}, visibility=visibility, allowed=allowed)
    metadata = {"dataset": config["dataset"], "fingerprint": fingerprint,
                "sources": [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files],
                "classes": classes, "label_mapping": originals, "clients": len(parts), "features": processor.feature_names,
                "preprocessor": processor.state_dict(), "class_counts": np.bincount(y, minlength=len(classes)),
                "split_counts": {s: np.bincount(y[idx], minlength=len(classes)) for s, idx in splits.items()},
                "row_order": "sorted file path, original CSV row order; splits index this order"}
    write_json(out / "metadata.json", metadata)
    write_json(out / "client_statistics.json", [{"client": i, "train": len(p["train"]), "validation": len(p["validation"]),
               "counts": np.bincount(y[p["train"]], minlength=len(classes))} for i, p in enumerate(parts)])
    from ..evaluation.plots import plot_partition
    plot_partition(y, parts, classes, out)
    for path, _ in chunks:
        path.unlink()  # exact files created by this invocation, no recursive deletion
    return PreparedData(out)
