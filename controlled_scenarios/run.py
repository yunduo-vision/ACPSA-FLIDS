import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import argparse
import copy
import numpy as np
from generate import generate
from acpsa_flids.utils.config import load_config, resolve
from acpsa_flids.federated.trainer import train
from acpsa_flids.utils.logger import write_json


def run(config, scenario, methods):
    output = resolve(config["output"])
    task_results = []
    for task in range(config["controlled"]["tasks"]):
        data = generate(config, scenario, task, output/f"data_{scenario}/task_{task:02d}")
        paired = {}
        for method in methods:
            c = copy.deepcopy(config)
            c["seed"] = data.metadata["seed"]
            c["aggregation"]["method"] = method
            # Joint protection is kept for classwise vs Bayesian; other weighting comparators are nonprivate.
            if method not in ("bayesian", "classwise"):
                c["variant"] = "fid_spa"
            c["output"] = str(output/f"scenario_{scenario}/task_{task:02d}/{method}")
            result = train(c, data)
            paired[method] = result
        if "classwise" in paired and "bayesian" in paired:
            base, ours = paired["classwise"]["unseen"]["per_client"], paired["bayesian"]["unseen"]["per_client"]
            gains = [b["macro_f1"]-a["macro_f1"] for a,b in zip(base,ours) if a["samples"]]
            task_results.append({"task": task, "paired_clients": len(gains), "gains": gains, "negative_gain_fraction": float(np.mean(np.array(gains)<0)) if gains else None,
                "client_scores": {"classwise": [r["macro_f1"] for r in base], "bayesian": [r["macro_f1"] for r in ours]},
                "class_support": np.sum(np.load(data.directory/"clients.npz")["visibility"], axis=0),
                "per_class_recall_gain": {name: paired["bayesian"]["detection"]["per_class"][name]["recall"]-paired["classwise"]["detection"]["per_class"][name]["recall"] for name in data.classes}})
        write_json(output/f"paired_{scenario}.json", task_results)
    return task_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/controlled_scenarios.yaml"); p.add_argument("--scenario", choices=["A", "B"], required=True)
    p.add_argument("--methods", nargs="+", default=["classwise", "bayesian"])
    a = p.parse_args(); run(load_config(a.config), a.scenario, a.methods)
