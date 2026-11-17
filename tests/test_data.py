import numpy as np
import pandas as pd
import pytest
from acpsa_flids.data.preprocessing import FeaturePreprocessor, numeric_frame
from acpsa_flids.data.label_mapping import map_labels
from acpsa_flids.data.partition import stratified_splits, partition_clients
from acpsa_flids.data.downloader import download_file, validate_csvs, extract_archive


def test_train_only_preprocessing():
    train = np.array([[1, np.nan, 1], [2, 4, 2], [3, 6, 3]])
    processor = FeaturePreprocessor().fit(train, ["bytes", "port", "duplicate"], ["bytes"])
    assert processor.keep.tolist() == [True, True, False]
    assert processor.medians[1] == 5
    old = processor.mean.copy()
    test = processor.transform([[1e9, np.inf, 1e9]])
    assert np.isfinite(test).all()
    np.testing.assert_array_equal(old, processor.mean)
    restored = FeaturePreprocessor.from_state(processor.state_dict())
    np.testing.assert_allclose(processor.transform(train), restored.transform(train))


def test_nonfinite_training_and_extreme_values():
    x = np.array([[np.inf,1],[2,2],[4,3],[np.nan,4]])
    processor = FeaturePreprocessor().fit(x,["bytes","port"],["bytes"])
    assert processor.medians[0] == 3
    assert np.isfinite(processor.transform(x)).all()


@pytest.mark.parametrize("dataset,raw,expected", [("cicids2017", ["BENIGN", "Web Attack – XSS", "Infiltration", "DoS Slowhttptest"], ["Benign", "Web Attack", "Infiltration", "DoS-Slowhttptest"]), ("nftoniotv2", ["Benign", "backdoor", "mitm", "ransomware"], ["Benign", "Backdoor", "MITM", "Ransomware"])])
def test_labels(dataset, raw, expected):
    assert map_labels(raw, dataset) == expected
    with pytest.raises(ValueError):
        map_labels(["new-unknown-label"], dataset)


def test_non_iid_and_split_disjointness():
    y = np.repeat(np.arange(10), 1000)
    splits = stratified_splits(y, {"full_test_fraction": .1, "regular_test_fraction": .2, "train_fraction": .2}, 42)
    all_idx = np.concatenate(list(splits.values()))
    assert len(np.unique(all_idx)) == len(y)
    clients, visible, allowed = partition_clients(y, splits["train"], 100, 10, 0, .1, 5, .1, 42)
    assert visible[:, 0].all() and visible.any(axis=0).all()
    assert (visible[:, 1:].sum(axis=1) <= 5).all()
    all_local = np.concatenate([np.concatenate(list(p.values())) for p in clients])
    np.testing.assert_array_equal(np.sort(all_local), np.sort(splits["train"]))
    repeat = partition_clients(y, splits["train"], 100, 10, 0, .1, 5, .1, 42)[0]
    for a, b in zip(clients, repeat):
        np.testing.assert_array_equal(a["train"], b["train"])


def test_csv_schema_cleaning():
    frame = pd.DataFrame({" Attack ": ["benign"], "IPV4_SRC_ADDR": ["1.2.3.4"], "L4_DST_PORT": [443], "PROTOCOL": [6], "bad": [np.inf]})
    numeric, labels = numeric_frame(frame, "Attack", ["^ipv4_.*addr$"])
    assert list(numeric.columns) == ["l4_dst_port", "protocol", "bad"]
    assert np.isnan(numeric["bad"][0])


def test_download_rejects_html(monkeypatch, tmp_path):
    class Response:
        status_code = 200
        headers = {"Content-Type": "text/html"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def raise_for_status(self): return None
    monkeypatch.setattr("requests.get", lambda *a, **kw: Response())
    with pytest.raises(RuntimeError, match="web page"):
        download_file("https://official.invalid", tmp_path/"archive.zip")


def test_zip_path_traversal(tmp_path):
    import zipfile
    archive = tmp_path/"bad.zip"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("../escaped.csv", "x")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path)
