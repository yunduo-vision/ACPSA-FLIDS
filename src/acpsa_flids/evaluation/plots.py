import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_partition(y, clients, classes, directory):
    matrix = np.array([np.bincount(y[p["train"]], minlength=len(classes)) for p in clients])
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.log1p(matrix), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right"); ax.set_ylabel("Client")
    fig.colorbar(im, ax=ax, label="log(1 + training samples)")
    fig.tight_layout(); fig.savefig(Path(directory)/"class_distribution.png", dpi=160); plt.close(fig)
    np.savetxt(Path(directory)/"visibility.csv", matrix > 0, delimiter=",", fmt="%d", header=",".join(classes))


def plot_history(path):
    path = Path(path)
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for key in ("macro_f1", "accuracy"):
        axes[0, 0].plot([r["round"] for r in rows], [r.get("validation", {}).get(key, np.nan) for r in rows], label=key)
    axes[0, 0].legend(); axes[0, 0].set_ylabel("Held-out regular test score")
    axes[0, 0].plot([r["round"] for r in rows], [r.get("unseen_macro_f1", np.nan) for r in rows], label="unseen_macro_f1")
    axes[0, 0].legend()
    axes[0, 1].plot([r["round"] for r in rows], [r["epsilon"] for r in rows]); axes[0, 1].set_ylabel("Ideal Gaussian cumulative epsilon")
    axes[1, 0].plot([r["round"] for r in rows], [r["support"] for r in rows]); axes[1, 0].set_ylabel("Privatized class support")
    axes[1, 1].plot([r["round"] for r in rows], [r["shrinkage"] for r in rows]); axes[1, 1].set_ylabel("Class shrinkage coefficient")
    for ax in axes.flat:
        ax.set_xlabel("Round"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path.parent/"convergence.png", dpi=160); plt.close(fig)
    private_summary = path.parent/"private_diagnostics/round_summary.jsonl"
    if private_summary.exists():
        local = [json.loads(s) for s in private_summary.read_text(encoding="utf-8").splitlines() if s.strip()]
        fig,ax = plt.subplots(figsize=(7,4))
        ax.plot([r["round"] for r in local],[r["weighted_local_training_loss"] for r in local],marker="o")
        ax.set_xlabel("Round"); ax.set_ylabel("Weighted local training loss (private)")
        fig.tight_layout(); fig.savefig(private_summary.parent/"local_loss.png",dpi=160); plt.close(fig)


def plot_confusion(matrix, classes, path):
    matrix = np.asarray(matrix)
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True)>0)
    fig, ax = plt.subplots(figsize=(max(7,len(classes)*.7), max(6,len(classes)*.6)))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    fig.colorbar(im,ax=ax,label="Row-normalized fraction")
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
