import _bootstrap
import argparse
from acpsa_flids.data.dataset import PreparedData
from acpsa_flids.federated.trainer import train
from acpsa_flids.utils.config import load_config, resolve

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cicids2017.yaml")
    p.add_argument("--resume"); p.add_argument("--output"); p.add_argument("--device")
    a = p.parse_args(); c = load_config(a.config)
    if a.output: c["output"] = a.output
    if a.device: c["device"] = a.device
    d = PreparedData(resolve(c["data"]["processed_dir"]))
    train(c, d, a.resume)
