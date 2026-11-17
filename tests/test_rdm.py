import pytest
from acpsa_flids.data.downloader import download_dataset


def test_official_rdm_open_api_resolution(monkeypatch, tmp_path):
    import zipfile
    import io
    import pandas as pd
    from acpsa_flids.data import downloader
    class Response:
        def __init__(self, data): self.data = data
        def raise_for_status(self): return None
        def json(self): return self.data
    def get(url, **kwargs):
        if url.endswith("/file-details"):
            return Response({"accessType": "OPEN", "name": "NF-ToN-IoT-v2"})
        assert url.endswith("/download")
        return Response({"canDownload": True, "url": "https://official-storage.invalid/payload"})
    monkeypatch.setattr("requests.get",get)
    def download(url,target,expected_sha256):
        assert url == "https://official-storage.invalid/payload"
        content = pd.DataFrame({**{f"F{j}":[j] for j in range(43)},"Attack":["Benign"]}).to_csv(index=False)
        with zipfile.ZipFile(target,"w") as z: z.writestr("NF-ToN-IoT-v2.csv",content)
        return "test-digest"
    monkeypatch.setattr(downloader,"download_file",download)
    files = download_dataset("nftoniotv2",tmp_path)
    assert len(files) == 1
    assert (tmp_path/"rdm_metadata.json").exists()
