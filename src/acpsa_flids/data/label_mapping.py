import re
import unicodedata
import yaml
from ..utils.config import ROOT


def normalize_label(value):
    text = unicodedata.normalize("NFKD", str(value)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def label_map(dataset):
    with (ROOT / "configs/labels.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)[dataset]


def map_labels(values, dataset):
    mapping = label_map(dataset)
    normalized = [normalize_label(v) for v in values]
    # CIC's Latin-1 corrupted dash can insert mojibake between Web Attack and subtype.
    normalized = [re.sub(r"^web attack .*?(brute force|sql injection|xss)$", r"web attack \1", v) for v in normalized]
    unknown = set(normalized) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped labels; update configs/labels.yaml: {sorted(unknown)}")
    return [mapping[v] for v in normalized]
