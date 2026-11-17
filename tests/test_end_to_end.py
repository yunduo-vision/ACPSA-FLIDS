import json
import numpy as np
import pandas as pd
import pytest
from acpsa_flids.utils.config import ROOT, load_config
from acpsa_flids.data.dataset import prepare, PreparedData
from acpsa_flids.federated.trainer import train
from acpsa_flids.federated.checkpoint import load_checkpoint
from acpsa_flids.inference.predictor import Predictor


@pytest.mark.parametrize("dataset", ["cicids2017", "nftoniotv2"])
def test_complete_real_schema_pipeline(tmp_path, dataset):
    # Generated unit-test records exercise both official schemas, never substitute real experiments.
    config = load_config(ROOT/f"configs/{dataset}.yaml")
    config["device"] = "cpu"
    config["federation"].update(clients=4, rounds=2, colluding_fraction=0, max_attack_classes=2)
    config["privacy"].update(h_min=3, epsilon=8, noise_rng="reproducible")
    config["local"].update(epochs=1, batch_size=32)
    config["model"]["channels"] = [2, 2, 4, 4, 4, 4]
    config["data"].update(raw_dir=str(tmp_path/"raw"), processed_dir=str(tmp_path/"processed"), train_fraction=.8, chunksize=300)
    config["output"] = str(tmp_path/"run")
    config["evaluation"].update(private_audit=True, attack_trees=5)
    raw = tmp_path/"raw"; raw.mkdir()
    labels = ["BENIGN", "Bot", "Infiltration"] if dataset == "cicids2017" else ["Benign", "backdoor", "mitm"]
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({config["data"]["label_column"]: np.repeat(labels, 100),
                          "duration": rng.lognormal(size=300), "bytes": rng.lognormal(size=300),
                          "L4_DST_PORT": rng.integers(0, 1024, 300), "PROTOCOL": rng.choice([6, 17], 300)})
    frame.loc[0, "bytes"] = np.inf
    frame.to_csv(raw/"records.csv", index=False)
    data = prepare(config)
    assert data.x.shape[0] == 300
    assert np.isfinite(data.x).all()
    assert prepare(config).classes == data.classes
    result = train(config, data)
    assert result["completed_rounds"] == 2
    assert result["epsilon"] == pytest.approx(8)
    checkpoint = tmp_path/"run/checkpoints/final_model.pt"
    predictor = Predictor(checkpoint, "cpu")
    prediction = predictor.predict(frame.head(4))
    assert len(prediction) == 4
    assert prediction.filter(like="probability_").sum(axis=1).to_numpy() == pytest.approx(np.ones(4))
    model, state = load_checkpoint(checkpoint, "cpu")
    assert state["accountant"]["releases"] == 2
    with pytest.raises(FileExistsError):
        train(config, data)
