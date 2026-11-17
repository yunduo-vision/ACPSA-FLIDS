import _bootstrap
import argparse
from acpsa_flids.data.downloader import download_dataset
from acpsa_flids.utils.config import load_config, resolve

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download/validate the official dataset release")
    p.add_argument("--dataset", required=True, choices=["cicids2017", "nftoniotv2"])
    p.add_argument("--raw-dir"); p.add_argument("--url"); p.add_argument("--sha256")
    args = p.parse_args()
    c = load_config(_bootstrap.ROOT/f"configs/{args.dataset}.yaml")
    files = download_dataset(args.dataset, resolve(args.raw_dir or c["data"]["raw_dir"]), args.url, args.sha256)
    print(f"Validated {len(files)} original CSV files")
