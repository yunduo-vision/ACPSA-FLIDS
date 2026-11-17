import copy
import math
from ..data.dataset import PreparedData
from ..utils.config import resolve
from ..utils.logger import write_json

VARIANTS = ["fid_spa", "indicator_dp", "update_dp", "full_sa", "acpsa"]
METHODS = ["global", "classwise", "confidence", "precision", "bayesian"]


def run_suite(config, kind="ablation"):
    from ..federated.trainer import train
    data = PreparedData(resolve(config["data"]["processed_dir"]))
    choices = config["experiments"]["budgets"] if kind == "privacy" else (VARIANTS if kind == "ablation" else METHODS)
    results = []
    for seed in config["experiments"]["seeds"]:
        for choice in choices:
            c = copy.deepcopy(config); c["seed"] = seed
            c["evaluation"]["private_audit"] = True
            if kind == "privacy":
                c["variant"], c["privacy"]["epsilon"], c["privacy"]["sigma"] = "acpsa", float(choice), None
            elif kind == "ablation":
                c["variant"] = choice
                c["aggregation"]["method"] = "bayesian" if choice == "acpsa" else "classwise"
            else:
                c["aggregation"]["method"] = choice
                # Identical plaintext inputs isolate aggregation from protection mechanisms.
                c["variant"] = "fid_spa"
            c["output"] = str(resolve(config["output"])/kind/f"{choice}_seed{seed}")
            result = train(c, data)
            results.append({"choice": str(choice), "seed": seed, "detection_macro_f1": result["detection"]["macro_f1"],
                            "unseen_macro_f1": result["unseen"]["client_mean"]["macro_f1"],
                            "attack_macro_f1": result.get("privacy_attack", {}).get("attack_macro_f1"),
                            "epsilon": result["epsilon"], "sigma": result["sigma"]})
            write_json(resolve(config["output"])/f"{kind}_results.json", results)
    if kind == "privacy":
        for row in results:
            reference = next(r for r in results if r["seed"] == row["seed"] and float(r["choice"]) == math.inf)
            row["utility_retention"] = row["detection_macro_f1"] / reference["detection_macro_f1"] if reference["detection_macro_f1"] else None
    import numpy as np
    summary = []
    for choice in choices:
        rows = [r for r in results if r["choice"] == str(choice)]
        summary.append({"choice": str(choice), "seeds": len(rows), **{k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows)>1 else 0.0}
                        for k in ("detection_macro_f1", "unseen_macro_f1", "attack_macro_f1") if all(r[k] is not None for r in rows)}})
    write_json(resolve(config["output"])/f"{kind}_results.json", results)
    write_json(resolve(config["output"])/f"{kind}_summary.json", summary)
    return results
