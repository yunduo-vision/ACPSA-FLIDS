import numpy as np
import torch


def blocks(model):
    feature = torch.cat([p.detach().reshape(-1).cpu() for p in model.features.parameters()]).double().numpy()
    head = torch.cat([model.classifier.weight.detach().cpu(), model.classifier.bias.detach().cpu()[:, None]], dim=1).double().numpy()
    return feature, head


def set_blocks(model, feature, head):
    with torch.no_grad():
        offset = 0
        for p in model.features.parameters():
            n = p.numel()
            p.copy_(torch.as_tensor(feature[offset:offset+n], dtype=p.dtype, device=p.device).view_as(p))
            offset += n
        if offset != len(feature) or head.shape != (model.classes, model.classifier.in_features + 1):
            raise ValueError("Parameter layout mismatch")
        model.classifier.weight.copy_(torch.as_tensor(head[:, :-1], dtype=model.classifier.weight.dtype, device=model.classifier.weight.device))
        model.classifier.bias.copy_(torch.as_tensor(head[:, -1], dtype=model.classifier.bias.dtype, device=model.classifier.bias.device))
