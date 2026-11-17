import _bootstrap
import argparse
import numpy as np
from acpsa_flids.data.dataset import PreparedData
from acpsa_flids.utils.logger import write_json


def check(data):
    splits = data.splits
    names = list(splits)
    for i,name in enumerate(names):
        if len(np.unique(splits[name])) != len(splits[name]):
            raise AssertionError(f"Duplicate row IDs in {name}")
        for other in names[i+1:]:
            if len(np.intersect1d(splits[name],splits[other],assume_unique=True)):
                raise AssertionError(f"Overlapping {name}/{other} splits")
    local = np.concatenate([np.concatenate(list(p.values())) for p in data.clients])
    if data.metadata["dataset"].startswith("controlled"):
        expected = np.concatenate([splits["train"],splits["regular"]])
    else:
        expected = splits["train"]
    np.testing.assert_array_equal(np.sort(local),np.sort(expected))
    invalid = 0
    for start in range(0,len(data.x),100000):
        invalid += int(np.sum(~np.isfinite(data.x[start:start+100000])))
    if invalid:
        raise AssertionError(f"{invalid} non-finite processed feature values")
    counts = np.array([np.bincount(data.y[p["train"]],minlength=len(data.classes)) for p in data.clients])
    if not (counts[:,0]>0).all() or not (counts.sum(axis=0)>0).all():
        raise AssertionError("Client/global coverage missing")
    result = {"rows": len(data.x), "features": data.x.shape[1], "classes": data.classes, "client_count": len(data.clients),
              "nonfinite_processed_values": invalid, "disjoint_splits": True, "disjoint_client_samples": True,
              "split_counts": {k:len(v) for k,v in splits.items()},
              "clients_observing_each_class": (counts>0).sum(axis=0),
              "max_attacks_observed_per_client": int((counts[:,1:]>0).sum(axis=1).max()),
              "class_counts_optimization": counts.sum(axis=0), "sample_conservation": True}
    write_json(data.directory/"quality_report.json",result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--processed-dir",required=True)
    a=p.parse_args(); print(check(PreparedData(a.processed_dir)))
