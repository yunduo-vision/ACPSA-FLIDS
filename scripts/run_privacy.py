import _bootstrap
import argparse
from acpsa_flids.utils.config import load_config
from acpsa_flids.evaluation.ablation import run_suite

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/privacy.yaml")
    run_suite(load_config(p.parse_args().config), "privacy")
