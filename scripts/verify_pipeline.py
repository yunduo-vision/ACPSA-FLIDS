"""Persistent integration evidence using explicitly marked schema fixtures."""
import _bootstrap
import numpy as np
import pandas as pd
from acpsa_flids.utils.config import load_config, resolve
from acpsa_flids.utils.logger import write_json
from acpsa_flids.data.dataset import prepare
from acpsa_flids.federated.trainer import train
from acpsa_flids.inference.predictor import Predictor


def verify():
    results = []
    root = resolve("outputs/verification/schema_fixtures")
    for dataset, labels in [("cicids2017", ["BENIGN", "Bot", "Infiltration"]), ("nftoniotv2", ["Benign", "backdoor", "mitm"])]:
        c = load_config(_bootstrap.ROOT/f"configs/{dataset}.yaml")
        c["federation"].update(clients=4, rounds=2, colluding_fraction=0, max_attack_classes=2)
        c["privacy"].update(h_min=3, epsilon=8.0, noise_rng="secure")
        c["local"]["epochs"] = 1
        c["data"].update(raw_dir=str(root/dataset/"raw"), processed_dir=str(root/dataset/"processed"), train_fraction=.8)
        c["output"] = str(root/dataset/"training")
        c["evaluation"].update(private_audit=True, attack_trees=10)
        rng = np.random.default_rng(987)
        count = 1200
        frame = pd.DataFrame({c["data"]["label_column"]: np.repeat(labels, count//len(labels)),
                              "Flow Duration": rng.lognormal(size=count), "Total Bytes": rng.lognormal(size=count),
                              "L4_DST_PORT": rng.integers(0, 1024, count), "PROTOCOL": rng.choice([6,17], count)})
        raw = resolve(c["data"]["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)
        csv = raw/"schema_fixture.csv"
        if not csv.exists():
            frame.to_csv(csv,index=False)
        write_json(root/dataset/"provenance.json", {"source": "generated software verification fixture", "real_dataset": False,
                   "purpose": "Verify schema processing, default six-convolution CNN, training, privacy, Shamir, checkpoint and inference"})
        data = prepare(c)
        result = train(c,data)
        checkpoint = resolve(c["output"])/"checkpoints/final_model.pt"
        predictor = Predictor(checkpoint)
        predicted = predictor.predict(frame.drop(columns=c["data"]["label_column"]).head(10))
        predicted.to_csv(root/dataset/"predictions.csv", index=False)
        results.append({"dataset_schema": dataset, "source": "generated test fixture, not a real IDS experiment", "completed_rounds": result["completed_rounds"],
                "epsilon": result["epsilon"], "joint_modules_executed": True, "checkpoint": str(checkpoint), "inference_rows": len(predicted)})
    write_json(root/"verification_summary.json", results)


if __name__ == "__main__":
    verify()
