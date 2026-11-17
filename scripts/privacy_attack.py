import _bootstrap
import argparse
from acpsa_flids.evaluation.privacy_attack import evaluate_attack

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--audit-dir", required=True); p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); print(evaluate_attack(a.audit_dir, a.seed))
