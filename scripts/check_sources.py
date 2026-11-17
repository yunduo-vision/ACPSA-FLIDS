"""Read-only source availability diagnostics. No credentials or fabricated registration."""
import _bootstrap
from concurrent.futures import ThreadPoolExecutor
import requests
from acpsa_flids.utils.logger import write_json

URLS = [
    "https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CSVs/MachineLearningCSV.zip",
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CSVs/MachineLearningCSV.zip",
    "https://rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5",
    "https://api.rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5",
    "https://api.rdm.uq.edu.au/datasets/a4ad7080-ef9c-11ed-a964-b70596e96ad5/file-details",
]


def check(url):
    try:
        with requests.get(url, stream=True, timeout=20) as response:
            return {"url": url, "status": response.status_code, "content_type": response.headers.get("Content-Type"),
                    "content_length": response.headers.get("Content-Length"), "redirected_to": response.url,
                    "first_bytes": next(response.iter_content(150), b"").decode("utf-8", errors="replace")}
    except requests.RequestException as error:
        return {"url": url, "error": str(error)}


if __name__ == "__main__":
    results = list(ThreadPoolExecutor(max_workers=4).map(check, URLS))
    write_json(_bootstrap.ROOT/"outputs/verification/source_availability.json", results)
    for row in results:
        print(row)
