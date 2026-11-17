import _bootstrap
import argparse
from pathlib import Path
from acpsa_flids.federated.checkpoint import load_checkpoint
from acpsa_flids.federated.trainer import evaluate
from acpsa_flids.data.dataset import PreparedData
from acpsa_flids.utils.logger import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--processed-dir")
    p.add_argument("--output"); p.add_argument("--device", default="auto")
    a = p.parse_args(); model, state = load_checkpoint(a.checkpoint, a.device)
    d = PreparedData(a.processed_dir or state["processed_dir"])
    if d.classes != state["classes"]:
        raise ValueError("Dataset/checkpoint class order mismatch")
    result = evaluate(model, d)
    out = a.output or str(Path(a.checkpoint).parent.parent/"evaluation_independent.json")
    write_json(out, result)
    print(f"Full-test accuracy={result['detection']['accuracy']:.6f} Macro-F1={result['detection']['macro_f1']:.6f}; {out}")
