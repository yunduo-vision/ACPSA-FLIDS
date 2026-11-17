import _bootstrap
import argparse
from acpsa_flids.utils.config import load_config
from acpsa_flids.evaluation.ablation import run_suite

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dataset", default="cicids2017", choices=["cicids2017", "nftoniotv2"])
    p.add_argument("--config"); p.add_argument("--kind", choices=["ablation", "aggregation"], default="ablation")
    a = p.parse_args(); run_suite(load_config(a.config or _bootstrap.ROOT/f"configs/{a.dataset}.yaml"), a.kind)
