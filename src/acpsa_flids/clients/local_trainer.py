import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def local_train(model, x, y, indices, config, seed):
    device = next(model.parameters()).device
    dataset = TensorDataset(torch.from_numpy(np.asarray(x[indices]).copy()), torch.from_numpy(np.asarray(y[indices], dtype=np.int64).copy()))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model.train()
    total, count = 0.0, 0
    for _ in range(config["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(yb)
            count += len(yb)
    return total / max(count, 1)
