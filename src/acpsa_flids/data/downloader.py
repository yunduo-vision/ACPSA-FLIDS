import hashlib
import re
import shutil
import zipfile
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd
import requests
import yaml
from ..utils.config import ROOT
from ..utils.logger import write_json


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def validate_csvs(directory, source):
    files = sorted(p for p in Path(directory).rglob("*.csv") if p.name not in source.get("exclude_files", []))
    if len(files) < source["expected_min_csvs"]:
        raise FileNotFoundError(f"Expected at least {source['expected_min_csvs']} original CSVs in {directory}; found {len(files)}")
    for file in files:
        columns = pd.read_csv(file, nrows=0, encoding="latin1").columns.str.strip()
        if source["label_column"].lower() not in columns.str.lower() or len(columns) < source["minimum_columns"]:
            raise ValueError(f"Wrong dataset format: {file}")
    return files


def download_file(url, target, expected_sha256=None, timeout=60):
    """Resume only a validated byte range; never accept an HTML landing page."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        actual = sha256(target)
        if expected_sha256 and actual.lower() != expected_sha256.lower():
            raise ValueError("Existing file checksum mismatch")
        return actual
    partial = target.with_suffix(target.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    with requests.get(url, headers={"Range": f"bytes={start}-"} if start else {}, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        if "html" in response.headers.get("Content-Type", "").lower():
            raise RuntimeError(f"Official endpoint returned a web page. Obtain the original archive from {url} and place it at {target}")
        if response.status_code == 206:
            match = re.match(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != start:
                raise RuntimeError("Invalid resume Content-Range")
            expected_total = int(match[3])
        else:
            start = 0
            expected_total = int(response.headers.get("Content-Length", 0))
        if expected_total and shutil.disk_usage(target.parent).free < (expected_total-start) * 1.1:
            raise OSError("Insufficient disk space for download")
        with partial.open("ab" if start else "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
    if expected_total and partial.stat().st_size != expected_total:
        raise IOError("Incomplete download; rerun to resume")
    actual = sha256(partial)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise ValueError("Downloaded checksum mismatch")
    partial.replace(target)
    return actual


def extract_archive(archive, directory):
    base = Path(directory).resolve()
    with zipfile.ZipFile(archive) as z:
        required = sum(i.file_size for i in z.infolist())
        if shutil.disk_usage(base).free < required * 1.1:
            raise OSError("Insufficient space for extraction; use --raw-dir on a larger disk")
        for item in z.infolist():
            dest = (base / item.filename).resolve()
            if not dest.is_relative_to(base):
                raise ValueError("Unsafe archive path")
        if z.testzip() is not None:
            raise ValueError("Archive CRC integrity check failed")
        z.extractall(base)


def download_dataset(dataset, directory, url=None, checksum=None):
    with (ROOT / "configs/sources.yaml").open(encoding="utf-8") as f:
        source = yaml.safe_load(f)[dataset]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        files = validate_csvs(directory, source)
    except FileNotFoundError:
        archive = directory / source["filename"]
        download_url = url or source["url"]
        if "rdm.uq.edu.au/files/" in download_url:
            uuid = urlparse(download_url).path.rstrip("/").split("/")[-1]
            # Follow the official public web application's own open-data API.
            api = f"https://api.rdm.uq.edu.au/datasets/{uuid}"
            details = requests.get(api+"/file-details", timeout=60)
            details.raise_for_status()
            meta = details.json()
            if meta.get("accessType") != "OPEN":
                raise RuntimeError("Official RDM dataset requires authorized interactive access")
            response = requests.get(api+"/download", timeout=60)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("canDownload") or not payload.get("url"):
                raise RuntimeError("RDM public file is not currently ready")
            download_url = payload["url"]
            write_json(directory/"rdm_metadata.json", meta)
        digest = download_file(download_url, archive, checksum or source["sha256"])
        if zipfile.is_zipfile(archive):
            extract_archive(archive, directory)
        else:
            # RDM can supply a direct CSV instead of an archive.
            first = pd.read_csv(archive, nrows=0, encoding="latin1")
            if len(first.columns) < source["minimum_columns"]:
                raise ValueError("Downloaded payload is neither the archive nor expected CSV")
            archive.replace(directory / (dataset + ".csv"))
        files = validate_csvs(directory, source)
    write_json(directory / "download_manifest.json", {"dataset": dataset, "landing": source["landing"],
               "files": [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
               "publisher_checksum_available": bool(checksum or source["sha256"])})
    return files
