"""Run configured real-data experiment families without inventing unavailable results."""
import _bootstrap
import argparse
from acpsa_flids.utils.config import load_config
from acpsa_flids.data.dataset import prepare
from acpsa_flids.evaluation.ablation import run_suite

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--suites", nargs="+", choices=["privacy", "ablation", "aggregation"], default=["privacy", "ablation", "aggregation"])
    a = p.parse_args(); config = load_config(a.config)
    prepare(config)
    for suite in a.suites:
        run_suite(config,suite)
