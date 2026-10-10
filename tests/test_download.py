import tarfile
import zipfile

import pytest

from upscaler.data.download import download_and_extract


def _zip(tmp_path):
    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("DIV2K_train_HR/0001.png", b"png-bytes")
    return src


def test_zip_download_extract_and_skip(tmp_path):
    src = _zip(tmp_path)
    raw = tmp_path / "raw"
    out = download_and_extract("DIV2K_train_HR", src.as_uri(), raw)
    assert (out / "DIV2K_train_HR" / "0001.png").read_bytes() == b"png-bytes"
    assert (out / ".done").exists()
    assert not (raw / "src.zip").exists()  # archive removed by default
    src.unlink()  # second call must not touch the URL
    assert download_and_extract("DIV2K_train_HR", src.as_uri(), raw) == out


def test_tar_extract(tmp_path):
    payload = tmp_path / "f.png"
    payload.write_bytes(b"abc")
    src = tmp_path / "Flickr2K.tar"
    with tarfile.open(src, "w") as t:
        t.add(payload, arcname="Flickr2K/Flickr2K_HR/000001.png")
    out = download_and_extract("Flickr2K", src.as_uri(), tmp_path / "raw", keep_archive=True)
    assert (out / "Flickr2K" / "Flickr2K_HR" / "000001.png").read_bytes() == b"abc"
    assert (tmp_path / "raw" / "Flickr2K.tar").exists()


def test_unknown_archive_type(tmp_path):
    src = tmp_path / "x.rar"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported archive"):
        download_and_extract("X", src.as_uri(), tmp_path / "raw")
