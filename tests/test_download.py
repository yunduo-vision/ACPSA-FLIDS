import hashlib
import pytest
from acpsa_flids.data.downloader import download_file


class Response:
    def __init__(self, code, payload, headers):
        self.status_code, self.payload, self.headers = code, payload, headers
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def raise_for_status(self): return None
    def iter_content(self, size): yield self.payload


def test_http_range_resume_and_checksum(monkeypatch, tmp_path):
    target = tmp_path/"payload.zip"
    target.with_suffix(".zip.part").write_bytes(b"abc")
    def get(url, headers, **kwargs):
        assert headers == {"Range": "bytes=3-"}
        return Response(206, b"def", {"Content-Range": "bytes 3-5/6", "Content-Type": "application/zip"})
    monkeypatch.setattr("requests.get", get)
    expected = hashlib.sha256(b"abcdef").hexdigest()
    assert download_file("https://official.invalid/archive", target, expected) == expected
    assert target.read_bytes() == b"abcdef"


def test_range_ignored_restarts(monkeypatch, tmp_path):
    target = tmp_path/"payload.zip"
    target.with_suffix(".zip.part").write_bytes(b"old")
    monkeypatch.setattr("requests.get", lambda *a, **k: Response(200, b"newfull", {"Content-Length": "7"}))
    download_file("https://official.invalid/archive", target)
    assert target.read_bytes() == b"newfull"


def test_bad_range_fails(monkeypatch, tmp_path):
    target = tmp_path/"payload.zip"
    target.with_suffix(".zip.part").write_bytes(b"abc")
    monkeypatch.setattr("requests.get", lambda *a, **k: Response(206, b"def", {"Content-Range": "bytes 2-4/6"}))
    with pytest.raises(RuntimeError):
        download_file("https://official.invalid/archive", target)
