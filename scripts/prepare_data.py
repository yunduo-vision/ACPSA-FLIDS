import _bootstrap
import argparse
from acpsa_flids.data.dataset import prepare
from acpsa_flids.utils.config import load_config

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cicids2017", "nftoniotv2"], default="cicids2017")
    p.add_argument("--config"); p.add_argument("--raw-dir"); p.add_argument("--processed-dir")
    a = p.parse_args()
    c = load_config(a.config or _bootstrap.ROOT/f"configs/{a.dataset}.yaml")
    if a.raw_dir: c["data"]["raw_dir"] = a.raw_dir
    if a.processed_dir: c["data"]["processed_dir"] = a.processed_dir
    d = prepare(c)
    print(f"Prepared {d.x.shape}, classes={d.classes}, cache={d.directory}")
