import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from ..utils.config import resolve
from ..utils.seed import seed_all, device_for
from ..utils.logger import append_json, write_json
from ..models.cnn import make_model
from ..models.model_utils import blocks
from ..aggregation.contribution import ContributionLayout
from ..aggregation.bayesian_shrinkage import PriorVariance
from ..aggregation.server import update_model
from ..privacy.rdp_accountant import RDPAccountant, calibrate_sigma
from ..privacy.gaussian_noise import check_honest, aggregate_variance
from ..secure_aggregation.protocol import SecureAggregation
from ..secure_aggregation.aggregation_committee import AggregationCommittee
from ..clients.client import FederatedClient, observation_features
from ..evaluation.detection_metrics import predict_probabilities, detection_metrics
from ..evaluation.unseen_classes import unseen_metrics
from ..evaluation.efficiency import EfficiencyMonitor
from ..evaluation.plots import plot_history, plot_confusion
from .checkpoint import save_checkpoint, load_checkpoint
from .communication import PlaintextAccumulator


def evaluate(model, data, split="full", batch_size=4096):
    idx = data.splits[split]
    predicted = predict_probabilities(model, data.x, idx, batch_size).argmax(axis=1)
    return {"detection": detection_metrics(data.y[idx], predicted, data.classes),
            "unseen": unseen_metrics(data.y[idx], predicted, data)}


def train(config, data, resume=None):
    seed_all(config["seed"], config["runtime"]["threads"])
    output = resolve(config["output"]); output.mkdir(parents=True, exist_ok=True)
    if not resume and ((output / "history.jsonl").exists() or (output / "privacy_ledger.jsonl").exists()):
        raise FileExistsError("Existing experiment history; use a new output or --resume")
    if len(data.clients) != config["federation"]["clients"]:
        raise ValueError("Cached partition client count differs from config")
    variant = config["variant"]
    protected = variant in ("acpsa", "indicator_dp", "update_dp")
    use_sa = variant in ("acpsa", "full_sa")
    if use_sa and not config["secure_aggregation"]["enabled"]:
        raise ValueError("ACPSA/Full-SA requires real threshold aggregation")
    model = make_model(data.x.shape[1], len(data.classes), config["model"]).to(device_for(config["device"]))
    feature, head = blocks(model)
    layout = ContributionLayout(len(feature), len(data.classes), head.shape[1], config["clipping"]["feature"], config["clipping"]["classes"])
    prior = PriorVariance(len(data.classes), config["aggregation"])
    accountant = RDPAccountant(config["privacy"]["delta"])
    limit = float(config["privacy"]["epsilon"]) if protected else math.inf
    sigma = (config["privacy"]["sigma"] if config["privacy"]["sigma"] is not None else calibrate_sigma(limit, config["federation"]["rounds"], config["privacy"]["delta"])) if protected else 0.0
    # Plaintext partial-DP ablations expose individual messages. Calibrate their
    # marginal noise to a local release, rather than claiming aggregate DP for them.
    accounting_sigma = sigma
    if variant in ("indicator_dp", "update_dp"):
        sigma *= math.sqrt(config["privacy"]["h_min"])
    config["privacy"]["sigma_resolved"] = sigma
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    write_json(output / "run_metadata.json", {"device": str(next(model.parameters()).device), "torch": torch.__version__,
               "joint_dimension": layout.dimension, "radius": layout.radius, "sensitivity": layout.sensitivity,
               "sigma": sigma, "variant": variant, "privacy_scope": "joint" if variant == "acpsa" else "partial or nonprivate ablation",
               "transport": "in-process independent committee objects; no network isolation",
               "rng": config["privacy"]["noise_rng"], "formal_bound": "ideal real-valued Gaussian RDP only"})
    start_round, best = 0, -math.inf
    if resume:
        model, state = load_checkpoint(resume, config["device"])
        if state["config"] != config or state["classes"] != data.classes or state.get("data_fingerprint") != data.metadata.get("fingerprint", data.metadata.get("generation_config")):
            raise ValueError("Resume requires identical resolved config and class mapping")
        a = state["accountant"]
        accountant = RDPAccountant(a["delta"], a["rho"], a["releases"])
        prior.values = np.asarray(state["prior_variance"])
        start_round, best = state["round"], state["best"]
        ledger = output / "privacy_ledger.jsonl"
        if ledger.exists():
            last = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
            if last["releases"] > accountant.releases:
                raise RuntimeError("A later aggregate was released after this checkpoint; refusing privacy-budget rollback")
    protocol = SecureAggregation(**config["secure_aggregation"]) if use_sa else None
    clients = [FederatedClient(i, data, config, layout) for i in range(len(data.clients))]
    total_communication = 0
    if resume and (output/"history.jsonl").exists():
        last_history = json.loads((output/"history.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        total_communication = last_history["total_communication_bytes"]
    completed = start_round
    save_checkpoint(output/"checkpoints/initial.pt", model, data, config, accountant, prior, start_round, best)
    for t in range(start_round, config["federation"]["rounds"]):
        if protected and not accountant.can_release(accounting_sigma, limit):
            write_json(output/"stop.json", {"reason": "next release exceeds budget", "round": t, "epsilon": accountant.epsilon})
            break
        monitor = EfficiencyMonitor()
        rng = np.random.default_rng(config["seed"]+t*1009)
        accepted = [i for i in range(len(clients)) if rng.random() >= config["federation"]["client_dropout"]]
        collusion_bound = math.floor(len(clients)*config["federation"]["colluding_fraction"])
        try:
            if not accepted:
                raise RuntimeError("No accepted clients")
            if protected and sigma > 0:
                check_honest(len(accepted), collusion_bound, config["privacy"]["h_min"])
            unavailable = config["federation"]["unavailable_nodes"]
            if protocol and protocol.nodes-len(set(unavailable)) < protocol.threshold:
                raise RuntimeError("Committee availability below threshold")
            committee = AggregationCommittee(protocol.nodes, layout.dimension, t) if protocol else PlaintextAccumulator(layout.dimension)
            attack_records, timings, diagnostics = [], [], []
            def diagnostic_sink(identity, stats):
                append_json(output/f"private_diagnostics/client_{identity:03d}.jsonl", stats)
                diagnostics.append((identity, stats))
            def attack_sink(identity, target, features):
                attack_records.append((identity, target.copy(), features))
            for identity in accepted:
                timings.append(clients[identity].execute(model, t, sigma, protocol, committee, len(accepted), diagnostic_sink,
                               attack_sink if config["evaluation"]["private_audit"] else None))
            start = time.perf_counter()
            shares = committee.finish(unavailable) if protocol else None
            aggregate = protocol.recover(shares, accepted, t) if protocol else committee.sum
            reconstruction = time.perf_counter()-start
        except (RuntimeError, OverflowError) as error:
            write_json(output/"abort.json", {"round": t, "reason": str(error), "released": False})
            raise
        # The release is irrevocable even if shrinkage/evaluation/checkpoint subsequently fails.
        accountant.step(accounting_sigma if protected else 0.0)
        append_json(output/"privacy_ledger.jsonl", {"round": t+1, **accountant.state_dict()})
        variance_sensitivity = layout.sensitivity if variant == "acpsa" else (2*np.sqrt(layout.classes) if variant == "indicator_dp" else 2*np.sqrt(layout.feature_bound**2+np.sum(layout.class_bounds**2)))
        variance = aggregate_variance(len(accepted), sigma, variance_sensitivity, config["privacy"]["h_min"])
        # Partial DP has no head noise for indicator-only; use the actual classifier variance.
        head_variance = 0.0 if variant == "indicator_dp" else variance
        stats = update_model(model, layout, aggregate, len(accepted), head_variance, prior, config["aggregation"])
        if attack_records:
            audit_dir = output/"private_audit"; audit_dir.mkdir(exist_ok=True)
            aggregate_features = observation_features(aggregate, layout)
            np.savez_compressed(audit_dir/f"round_{t+1:04d}.npz", clients=[r[0] for r in attack_records],
                targets=np.stack([r[1] for r in attack_records]),
                observations=np.stack([aggregate_features if r[2] is None else r[2] for r in attack_records]))
        validation = {}
        if (t+1) % config["evaluation"]["every"] == 0:
            # This held-out regular split selects checkpoints; D_full is reserved for final scoring.
            idx = data.splits["regular"]
            predicted = predict_probabilities(model, data.x, idx, config["evaluation"]["batch_size"]).argmax(axis=1)
            validation = detection_metrics(data.y[idx], predicted, data.classes)
            unseen = unseen_metrics(data.y[idx], predicted, data)
            stats["unseen_macro_f1"] = unseen["client_mean"]["macro_f1"]
        communication = sum(s["upload_bytes"] for s in timings)
        communication += len(accepted)*sum(p.numel()*p.element_size() for p in model.parameters())
        communication += sum(s.value.nbytes for s in shares) if shares else 0
        total_communication += communication
        record = {"round": t+1, "participants": len(accepted), "collusion_bound": collusion_bound,
                  "unknown_noise_contributor_lower_bound": len(accepted)-collusion_bound,
                  "nominal_noise_variance": variance, "honest_noise_variance_lower_bound": aggregate_variance(len(accepted)-collusion_bound, sigma, variance_sensitivity, config["privacy"]["h_min"]),
                  "client_noise_std": sigma*variance_sensitivity/math.sqrt(config["privacy"]["h_min"]),
                  "epsilon": accountant.epsilon, "remaining_epsilon": max(0, limit-accountant.epsilon) if math.isfinite(limit) else None,
                  "validation": validation, "reconstruction_seconds": reconstruction,
                  "client_protection_seconds": sum(s["protection_seconds"] for s in timings),
                  "sharing_seconds": sum(s["sharing_seconds"] for s in timings),
                  "node_sum_seconds": sum(s["node_sum_seconds"] for s in timings),
                  "communication_bytes": communication, "total_communication_bytes": total_communication,
                  **stats, **monitor.finish()}
        append_json(output/"history.jsonl", record)
        append_json(output/"private_diagnostics/round_summary.jsonl", {
            "round": t+1,
            "weighted_local_training_loss": np.average([s["loss"] for _,s in diagnostics], weights=[len(data.clients[i]["train"]) for i,_ in diagnostics]),
            "mean_feature_clip_ratio": np.mean([s["clipping"]["feature"]["ratio"] for _,s in diagnostics]),
            "mean_class_clip_ratios": np.mean([[b["ratio"] for b in s["clipping"]["classes"]] for _,s in diagnostics],axis=0),
            "visibility": "trusted researcher local diagnostics only, outside public DP transcript"})
        completed = t+1
        improved = validation.get("macro_f1", -math.inf) > best
        if improved:
            best = validation["macro_f1"]
            save_checkpoint(output/"checkpoints/best_model.pt", model, data, config, accountant, prior, completed, best)
        if completed % config["runtime"]["checkpoint_every"] == 0:
            save_checkpoint(output/"checkpoints/latest.pt", model, data, config, accountant, prior, completed, best)
        print(f"round={completed} clients={len(accepted)} regular_f1={validation.get('macro_f1', float('nan')):.5f} epsilon={accountant.epsilon:.6g}", flush=True)
    save_checkpoint(output/"checkpoints/final_model.pt", model, data, config, accountant, prior, completed, best)
    final = evaluate(model, data, batch_size=config["evaluation"]["batch_size"])
    final["minority"] = {c: final["detection"]["per_class"][c] for c in config["evaluation"]["minority_classes"] if c in data.classes}
    final["completed_rounds"], final["epsilon"], final["sigma"] = completed, accountant.epsilon, sigma
    write_json(output/"evaluation_final.json", final)
    plot_confusion(final["detection"]["confusion_matrix"], data.classes, output/"confusion_matrix.png")
    if (output/"checkpoints/best_model.pt").exists():
        best_model, _ = load_checkpoint(output/"checkpoints/best_model.pt", config["device"])
        write_json(output/"evaluation_best.json", evaluate(best_model, data, batch_size=config["evaluation"]["batch_size"]))
    if config["evaluation"]["private_audit"] and completed:
        from ..evaluation.privacy_attack import evaluate_attack
        if len(clients) >= 4:
            final["privacy_attack"] = evaluate_attack(output/"private_audit", config["seed"], config["evaluation"]["attack_test_fraction"], config["evaluation"]["attack_trees"], config["evaluation"]["attack_max_depth"], config["evaluation"]["attack_min_samples_leaf"])
            write_json(output/"evaluation_final.json", final)
    if completed:
        plot_history(output/"history.jsonl")
    return final
