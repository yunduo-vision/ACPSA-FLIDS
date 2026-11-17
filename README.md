# ACPSA-FLIDS
Attack-Class-Privacy-aware Secure Aggregation for Federated Intrusion Detection Systems

## Environment Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

## Data Preparation

```bash
python scripts/download_data.py --dataset cicids2017
python scripts/download_data.py --dataset nftoniotv2
python scripts/prepare_data.py --dataset cicids2017
python scripts/prepare_data.py --dataset nftoniotv2
```

Official dataset sources:
[CIC-IDS2017 MachineLearningCSV](https://www.unb.ca/cic/datasets/ids-2017.html)
[UQ NetFlow V2](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)

```bash
python scripts/download_data.py --dataset nftoniotv2 --url "AUTHORIZED_DIRECT_FILE_URL" --raw-dir "D:/ACPSA_FLIDS_data/raw/nftoniotv2"
```

Alternatively, manually place the official CIC-IDS2017 CSV files in `data/raw/cicids2017/` (subdirectories are supported), or place `NF-ToN-IoT-v2.csv` in `data/raw/nftoniotv2/`. Use `--sha256` to supply a known checksum provided by the publisher.


```bash
python scripts/prepare_data.py --config configs/nftoniotv2.yaml --raw-dir "D:/ACPSA_FLIDS_data/raw/nftoniotv2" --processed-dir "D:/ACPSA_FLIDS_data/processed/nftoniotv2"
```


## Training, Resuming, Evaluation, and Inference

```bash
python scripts/train.py --config configs/cicids2017.yaml
python scripts/train.py --config configs/nftoniotv2.yaml
python scripts/run_real_experiment.py --config configs/nftoniotv2_local.yaml
python scripts/train.py --config configs/cicids2017.yaml --resume outputs/cicids2017/checkpoints/latest.pt
python scripts/evaluate.py --checkpoint outputs/cicids2017/checkpoints/best_model.pt
python scripts/inference.py --checkpoint outputs/cicids2017/checkpoints/best_model.pt --input data/new_flows.csv --output outputs/predictions.csv
```


## Experiment Entry Points

```bash
python scripts/run_privacy.py --config configs/privacy.yaml
python scripts/run_ablation.py --dataset cicids2017
python scripts/run_ablation.py --dataset nftoniotv2
python scripts/run_ablation.py --dataset cicids2017 --kind aggregation
python scripts/run_sensitivity.py --config configs/cicids2017.yaml --parameter clipping
python scripts/run_sensitivity.py --config configs/cicids2017.yaml --parameter alpha
python scripts/run_full_experiments.py --config configs/nftoniotv2_local.yaml --suites privacy ablation aggregation
python scripts/privacy_attack.py --audit-dir outputs/cicids2017/private_audit
python scripts/run_efficiency.py --config configs/cicids2017.yaml
```

Privacy experiments use epsilon values of 0.5, 1, 2, 4, 8, and infinity, automatically calibrate sigma, and save detection F1, locally unseen-class F1, attack F1, utility retention, noise parameters, and the actual composed privacy budget. Results are summarized as means and standard deviations across three seeds. Protection ablations include FID-SPA, Indicator-DP, Update-only DP, Full-SA, and ACPSA. Aggregation comparisons include global FedAvg with uniform client weights, class-wise averaging, confidence weighting, precision weighting, and Bayesian shrinkage; see the assumptions documentation for implementation definitions and differences in protection scope. The attacker uses only observations permitted by its threat model. Supervised training and test sets are separated by client, keeping all rounds from a given client in the same split.

`run_efficiency.py` is a protocol microbenchmark that performs actual secret-sharing computations for K = 10, 20, 50, 100, 150, and 200, including client and node dropout. **It does not represent a complete system experiment that includes local optimization and a real network.** The default h_min = 80 is not automatically reduced for smaller client populations; the protocol aborts when the noise-contributor condition fails. Training logs record end-to-end training time, communication for model broadcasts and all shares, peak GPU memory, and CPU and GPU utilization. Client dropout and node failures are controlled through the federation configuration.

## Separate Controlled Scenarios from the Paper

```bash
python controlled_scenarios/run.py --scenario A
python controlled_scenarios/run.py --scenario B
python scripts/analyze_results.py --directory outputs/controlled
python controlled_scenarios/shrinkage_analysis.py
python controlled_scenarios/shrinkage_analysis.py --paired-file outputs/controlled/paired_A.json
```

The full default settings are 8 or 10 classes, 20-dimensional features, 100 clients with 1,000 samples each, 500 test samples per class, 20 tasks, a 90:10 local training/validation split, and alpha = 0.1. Within each task, methods share the data, initialization, and local training settings. The default comparison evaluates class-wise averaging and Bayesian shrinkage under the same DP + SA configuration. The fixed-support Gaussian estimation experiment uses a noise standard deviation of 10 and 3,000 draws per support level. Recall gains are not plotted without actual paired results. `configs/verification_controlled.yaml` is intended only for software verification with 4 clients, 2 rounds, and 1 task.

## Validation and Privacy Boundaries

```bash
python -m pytest -q --junitxml=outputs/verification/pytest.xml
python scripts/check_sources.py
python scripts/verify_pipeline.py
python controlled_scenarios/run.py --config configs/verification_controlled.yaml --scenario A
python controlled_scenarios/run.py --config configs/verification_controlled.yaml --scenario B
```
