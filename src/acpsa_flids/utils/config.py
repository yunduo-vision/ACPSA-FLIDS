from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[3]


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parent = config.pop("extends", None)
    return merge(load_config(path.parent / parent), config) if parent else config


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
