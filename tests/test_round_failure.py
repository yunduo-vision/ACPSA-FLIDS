import sys
from pathlib import Path
import json
import copy
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"controlled_scenarios"))
from generate import generate
from acpsa_flids.utils.config import ROOT, load_config
from acpsa_flids.federated.trainer import train


@pytest.mark.parametrize("failure", ["honest", "committee"])
def test_round_aborts_before_release(tmp_path, failure):
    c = load_config(ROOT/"configs/verification_controlled.yaml")
    c["output"] = str(tmp_path/"run")
    data = generate(c, "A", 0, tmp_path/"data")
    if failure == "honest":
        c["privacy"]["h_min"] = 80
    else:
        c["federation"]["unavailable_nodes"] = [1,2,3]
    with pytest.raises(RuntimeError):
        train(c,data)
    assert not (tmp_path/"run/privacy_ledger.jsonl").exists()
    assert json.loads((tmp_path/"run/abort.json").read_text())["released"] is False


def test_resume_keeps_completed_budget_and_refuses_rollback(tmp_path):
    c = load_config(ROOT/"configs/verification_controlled.yaml")
    c["output"] = str(tmp_path/"run")
    data = generate(c,"A",0,tmp_path/"data")
    train(c,data)
    state_path = tmp_path/"run/checkpoints/final_model.pt"
    resumed = train(c,data,resume=state_path)
    assert resumed["completed_rounds"] == 2
    with (tmp_path/"run/privacy_ledger.jsonl").open("a") as f:
        f.write('{"releases":3}\n')
    with pytest.raises(RuntimeError, match="rollback"):
        train(c,data,resume=state_path)
