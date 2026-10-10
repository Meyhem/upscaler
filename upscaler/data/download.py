"""Download and extract the DF2K training data and DIV2K validation data.

python -m upscaler.data.download [--raw-dir datasets/raw] [--only DIV2K_valid_HR] [--url NAME=URL]
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

URLS = {
    "DIV2K_train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "Flickr2K": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
}


def _fetch(url: str, out: Path) -> None:
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def _extract(archive: Path, dest: Path) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)  # zipfile strips absolute paths and '..'
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")


def download_and_extract(name: str, url: str, raw_dir: str | Path, keep_archive: bool = False) -> Path:
    raw_dir = Path(raw_dir)
    dest = raw_dir / name
    marker = dest / ".done"
    if marker.exists():
        print(f"{name}: already present, skipping")
        return dest
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / url.rsplit("/", 1)[-1]
    if not archive.exists():
        part = archive.with_name(archive.name + ".part")
        _fetch(url, part)
        part.rename(archive)
    dest.mkdir(parents=True, exist_ok=True)
    _extract(archive, dest)
    marker.touch()
    if not keep_archive:
        archive.unlink()
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="datasets/raw")
    ap.add_argument("--only", nargs="*", choices=sorted(URLS), help="subset of datasets")
    ap.add_argument("--url", action="append", default=[], metavar="NAME=URL", help="override a mirror URL")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args(argv)
    urls = dict(URLS)
    for item in args.url:
        key, _, value = item.partition("=")
        urls[key] = value
    failed = []
    for name in args.only or list(URLS):
        try:
            download_and_extract(name, urls[name], args.raw_dir, args.keep_archives)
        except OSError as e:
            print(f"{name}: download failed ({e}). Retry later or pass --url {name}=<mirror>", file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
