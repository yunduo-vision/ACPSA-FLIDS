import _bootstrap
import argparse
import copy
import math
import time
import numpy as np
from acpsa_flids.utils.config import load_config, resolve
from acpsa_flids.utils.logger import write_json
from acpsa_flids.models.cnn import make_model
from acpsa_flids.models.model_utils import blocks
from acpsa_flids.aggregation.contribution import ContributionLayout
from acpsa_flids.privacy.gaussian_noise import check_honest, client_noise, private_rng
from acpsa_flids.privacy.rdp_accountant import calibrate_sigma
from acpsa_flids.secure_aggregation.protocol import SecureAggregation
from acpsa_flids.secure_aggregation.aggregation_committee import AggregationCommittee


def protocol_benchmark(config):
    # Measures actual protocol operations, without asserting any IDS detection result.
    model = make_model(43, 10, config["model"])
    f,h = blocks(model)
    layout = ContributionLayout(len(f), 10, h.shape[1], config["clipping"]["feature"], config["clipping"]["classes"])
    protocol = SecureAggregation(**config["secure_aggregation"])
    sigma = calibrate_sigma(config["privacy"]["epsilon"], config["federation"]["rounds"], config["privacy"]["delta"])
    rng = np.random.default_rng(config["seed"])
    records = []
    for k in config["experiments"]["clients"]:
        for dropout in config["experiments"]["dropout_rates"]:
            for failed_nodes in range(protocol.nodes+1):
                m = k-round(k*dropout)
                row = {"clients": k, "accepted": m, "client_dropout": dropout, "failed_nodes": failed_nodes, "h_min": config["privacy"]["h_min"]}
                try:
                    check_honest(m, math.floor(k*config["federation"]["colluding_fraction"]), config["privacy"]["h_min"])
                    if protocol.nodes-failed_nodes < protocol.threshold:
                        raise RuntimeError("Committee availability below threshold")
                    committee = AggregationCommittee(protocol.nodes, layout.dimension)
                    start = time.perf_counter()
                    sharing, node_sum, communication = 0., 0., 0
                    for i in range(m):
                        v, _ = layout.build(rng.normal(size=layout.feature_dim), rng.normal(size=(10,layout.head_dim)), rng.integers(0,2,10))
                        v += client_noise(layout.dimension, sigma, layout.sensitivity, config["privacy"]["h_min"], private_rng(config["privacy"]["noise_rng"], config["seed"]+i))
                        t = time.perf_counter(); shares = protocol.share(v, m); sharing += time.perf_counter()-t
                        t = time.perf_counter(); committee.deliver(i,shares); node_sum += time.perf_counter()-t
                        communication += sum(s.nbytes for s in shares.values())
                    t = time.perf_counter(); aggregate = protocol.recover(committee.finish(range(1,failed_nodes+1)), range(m)); reconstruct_time = time.perf_counter()-t
                    row.update(status="released", protocol_seconds=time.perf_counter()-start, sharing_seconds=sharing, node_sum_seconds=node_sum,
                               reconstruction_seconds=reconstruct_time, communication_bytes=communication+(protocol.nodes-failed_nodes)*layout.dimension*8,
                               aggregate_l2=float(np.linalg.norm(aggregate)))
                except (RuntimeError, OverflowError) as e:
                    row.update(status="aborted_before_release", reason=str(e))
                records.append(row)
                write_json(resolve(config["output"])/"protocol_efficiency.json", {"scope": "actual Shamir protocol microbenchmark on bounded numeric workloads; excludes local training and network latency", "results": records})
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/cicids2017.yaml")
    a = p.parse_args(); protocol_benchmark(load_config(a.config))
