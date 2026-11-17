import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acpsa_flids.utils.logger import write_json


def analyze(directory):
    root = Path(directory)
    privacy_file = root/"privacy_results.json"
    if privacy_file.exists():
        rows = json.loads(privacy_file.read_text(encoding="utf-8"))
        budgets = sorted(set(float(r["choice"]) for r in rows))
        fig, axes = plt.subplots(1,3,figsize=(13,4))
        for ax, key in zip(axes,["detection_macro_f1","unseen_macro_f1","attack_macro_f1"]):
            means, stds = [], []
            for budget in budgets:
                values = [r[key] for r in rows if float(r["choice"]) == budget and r[key] is not None]
                means.append(np.mean(values) if values else np.nan)
                stds.append(np.std(values,ddof=1) if len(values)>1 else 0)
            ax.errorbar(range(len(budgets)),means,yerr=stds,marker="o",capsize=3)
            ax.set_xticks(range(len(budgets)),[f"{b:g}" for b in budgets]); ax.set_xlabel("Target epsilon")
            ax.set_ylabel(key); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(root/"privacy_utility.png",dpi=180); plt.close(fig)
    paired_files = sorted(root.glob("paired_*.json"))
    for path in paired_files:
        tasks = json.loads(path.read_text(encoding="utf-8"))
        gains = np.concatenate([t["gains"] for t in tasks]) if tasks else np.array([])
        if not len(gains):
            continue
        summary = {"independent_tasks": len(tasks), "nested_paired_clients": len(gains), "negative_gain_fraction": float(np.mean(gains<0)),
                   "gain_mean": float(gains.mean()), "gain_quantiles": np.quantile(gains,[.1,.5,.9]),
                   "inference_unit": "tasks; clients are nested and not independent tasks"}
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        for method in ("classwise", "bayesian"):
            scores = np.array([s for t in tasks for s in t["client_scores"][method] if s is not None])
            scores.sort()
            axes[0].step(scores,np.arange(1,len(scores)+1)/len(scores),label=method)
            summary[method] = {"worst_10_percent_mean": float(scores[:max(1,int(np.ceil(len(scores)*.1)))].mean()), "quantile_10": float(np.quantile(scores,.1)), "median": float(np.median(scores))}
        axes[0].set_xlabel("Client unseen Macro-F1"); axes[0].set_ylabel("Empirical CDF"); axes[0].legend()
        axes[1].hist(gains,bins=min(30,max(3,len(gains)//5))); axes[1].axvline(0,color="black",linestyle="--")
        axes[1].set_xlabel("Paired unseen-F1 gain"); axes[1].set_ylabel("Nested client count")
        fig.tight_layout(); fig.savefig(path.with_suffix(".png"),dpi=180); plt.close(fig)
        write_json(path.with_name(path.stem+"_summary.json"),summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--directory", required=True)
    analyze(p.parse_args().directory)
