from pathlib import Path
import torch
from ..models.cnn import make_model
from ..utils.seed import device_for


def save_checkpoint(path, model, data, config, accountant, prior, round_id, best):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "model_config": config["model"], "input_dim": model.input_dim, "classes": data.classes,
               "preprocessor": data.preprocessor_state, "label_mapping": data.metadata.get("label_mapping", {}),
               "config": config, "accountant": accountant.state_dict(), "prior_variance": prior.values,
               "round": round_id, "best": best, "processed_dir": str(data.directory.resolve()),
               "data_fingerprint": data.metadata.get("fingerprint", data.metadata.get("generation_config"))}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def load_checkpoint(path, device="auto"):
    device = device_for(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(state["input_dim"], len(state["classes"]), state["model_config"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state
