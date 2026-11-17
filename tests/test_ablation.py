import numpy as np
import pytest
from acpsa_flids.utils.config import ROOT, load_config
from acpsa_flids.data.dataset import prepare
from acpsa_flids.federated.trainer import train


@pytest.mark.parametrize("variant,method", [("fid_spa", "classwise"), ("fid_spa", "global"), ("fid_spa", "confidence"), ("fid_spa", "precision"), ("full_sa", "classwise"), ("indicator_dp", "classwise"), ("update_dp", "classwise")])
def test_ablation_actual_training(tmp_path, variant, method):
    import pandas as pd
    config = load_config(ROOT/"configs/cicids2017.yaml")
    config["variant"] = variant
    config["aggregation"]["method"] = method
    config["model"]["channels"] = [2]*6
    config["federation"].update(clients=4, rounds=1, colluding_fraction=0)
    config["privacy"].update(h_min=3, epsilon=8, noise_rng="reproducible")
    config["local"]["epochs"] = 1
    config["device"] = "cpu"
    config["data"].update(raw_dir=str(tmp_path/"raw"), processed_dir=str(tmp_path/"processed"), train_fraction=.8)
    config["output"] = str(tmp_path/"run")
    raw = tmp_path/"raw"; raw.mkdir()
    rng = np.random.default_rng(4)
    pd.DataFrame({"Label": np.repeat(["BENIGN", "Bot"], 100), "Flow Duration": rng.lognormal(size=200), "Bytes": rng.random(200)}).to_csv(raw/"records.csv",index=False)
    result = train(config, prepare(config))
    assert result["completed_rounds"] == 1
    if variant in ("indicator_dp", "update_dp"):
        assert result["epsilon"] == pytest.approx(8)


def test_budget_ledger_prevents_restart(tmp_path):
    from types import SimpleNamespace
    config = load_config(ROOT/"configs/default.yaml")
    config["output"] = str(tmp_path)
    (tmp_path/"privacy_ledger.jsonl").write_text('{"releases":1}\n')
    with pytest.raises(FileExistsError):
        train(config, SimpleNamespace())
