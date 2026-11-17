import _bootstrap
import argparse
from pathlib import Path
import pandas as pd
from acpsa_flids.inference.predictor import Predictor

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/predictions.csv"); p.add_argument("--device", default="auto")
    p.add_argument("--encoding", default="latin1"); p.add_argument("--chunksize", type=int, default=100000)
    a = p.parse_args(); predictor = Predictor(a.checkpoint, a.device)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for frame in pd.read_csv(a.input, chunksize=a.chunksize, encoding=a.encoding, low_memory=False):
        predictor.predict(frame).to_csv(out, mode="w" if first else "a", header=first, index=False)
        first = False
    print(out.resolve())
