import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acpsa_flids.utils.config import load_config, resolve
from acpsa_flids.utils.logger import write_json


def analyze(config, paired_file=None):
    c = config["controlled"]
    rng = np.random.default_rng(config["seed"])
    supports = np.array(c["supports"])
    nu, s2, draws = c["gaussian_noise_std"], c["gaussian_prior_variance"], c["gaussian_draws"]
    rows = []
    for n in supports:
        # Fixed true mean makes frequentist bias^2 and variance identifiable.
        truth = np.sqrt(s2)
        observations = truth + rng.normal(0, nu/n, draws)
        weight = n*n*s2/(n*n*s2+nu*nu)
        estimate = weight*observations
        prior_truth = rng.normal(0, np.sqrt(s2), draws)
        prior_observation = prior_truth + rng.normal(0, nu/n, draws)
        rows.append({"support": n, "lambda": weight, "draws": draws, "direct_mse": np.mean((observations-truth)**2),
                     "shrinkage_mse": np.mean((estimate-truth)**2), "bias_squared": (estimate.mean()-truth)**2,
                     "variance": estimate.var(), "prior_averaged_mse": np.mean((weight*prior_observation-prior_truth)**2),
                     "ideal_bayes_risk": s2*(nu/n)**2/(s2+(nu/n)**2)})
    out = resolve(config["output"])/"shrinkage_analysis"; out.mkdir(parents=True, exist_ok=True)
    write_json(out/"gaussian_estimation.json", {"noise_std": nu, "seed": config["seed"], "results": rows})
    fig, axes = plt.subplots(2,2,figsize=(11,8))
    for std in [nu/2, nu, nu*2]:
        axes[0,0].plot(supports, supports**2*s2/(supports**2*s2+std**2), marker="o", label=f"std={std:g}")
    axes[0,0].set_ylabel("Shrinkage coefficient"); axes[0,0].legend()
    for key in ["direct_mse", "shrinkage_mse", "ideal_bayes_risk"]:
        axes[0,1].plot(supports, [r[key] for r in rows], marker="o", label=key)
    axes[0,1].set_yscale("log"); axes[0,1].legend()
    for key in ["bias_squared", "variance"]:
        axes[1,0].plot(supports, [r[key] for r in rows], marker="o", label=key)
    axes[1,0].legend()
    if paired_file:
        tasks = json.loads(Path(paired_file).read_text(encoding="utf-8"))
        for task in tasks:
            axes[1,1].scatter(task["class_support"], list(task["per_class_recall_gain"].values()), alpha=.4)
        axes[1,1].set_ylabel("Measured recall gain (Bayesian - classwise)")
    else:
        axes[1,1].text(.5,.5,"Requires measured paired controlled-task results", ha="center", va="center", wrap=True)
        axes[1,1].set_axis_off()
    for ax in axes.flat:
        ax.set_xlabel("Class support"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/"shrinkage.png", dpi=180); plt.close(fig)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/controlled_scenarios.yaml"); p.add_argument("--paired-file")
    a = p.parse_args(); analyze(load_config(a.config), a.paired_file)
