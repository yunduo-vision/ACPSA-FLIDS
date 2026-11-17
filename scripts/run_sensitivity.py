import _bootstrap
import argparse
import copy
from acpsa_flids.utils.config import load_config, resolve
from acpsa_flids.utils.logger import write_json
from acpsa_flids.data.dataset import PreparedData, prepare
from acpsa_flids.federated.trainer import train

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/cicids2017.yaml")
    p.add_argument("--parameter", choices=["clipping", "alpha"], required=True)
    a = p.parse_args(); config = load_config(a.config); rows = []
    values = config["experiments"]["clipping_values" if a.parameter == "clipping" else "alphas"]
    for value in values:
        c = copy.deepcopy(config); c["output"] = str(resolve(config["output"])/f"{a.parameter}_{value}")
        if a.parameter == "clipping":
            c["clipping"].update(feature=value, classes=value/10)
            data = PreparedData(resolve(c["data"]["processed_dir"]))
        else:
            c["federation"]["alpha"] = value
            c["data"]["processed_dir"] += f"_alpha_{value}"
            data = prepare(c)
        result = train(c,data)
        rows.append({"value": value, "macro_f1": result["detection"]["macro_f1"], "sigma": result["sigma"]})
        write_json(resolve(config["output"])/f"sensitivity_{a.parameter}.json", rows)
