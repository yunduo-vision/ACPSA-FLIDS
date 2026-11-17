import torch
from .threshold_selection import select_threshold


@torch.no_grad()
def probe(model, observed, count, candidates, generator):
    model.eval()
    probes = torch.randn(count, model.input_dim, generator=generator).to(next(model.parameters()).device)
    confidence = model(probes).softmax(dim=1).mean(dim=0).cpu().double().numpy()
    threshold, indicator = select_threshold(confidence, observed, candidates)
    return indicator, confidence, threshold
