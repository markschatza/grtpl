from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm.auto import tqdm


HC3_FILE_BASE = "https://crcns.org/files/data/hc3/"
HC3_NERSC_BASE = "https://portal.nersc.gov/project/crcns/download/hc-3/"


PUBLIC_HC3_FILES = {
    "data_description": "crcns-hc3-data-description.pdf",
    "metadata_tables": "crcns-hc3-metadata-tables.zip",
    "channel_order": "crcns-hc3-channelorder.zip",
}


@dataclass(frozen=True)
class DownloadedFile:
    name: str
    url: str
    path: Path
    bytes_written: int


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> DownloadedFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    written = 0
    with destination.open("wb") as handle:
        with tqdm(total=total or None, unit="B", unit_scale=True, desc=destination.name) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                progress.update(len(chunk))

    return DownloadedFile(destination.name, url, destination, written)


def download_public_hc3_docs(raw_dir: Path) -> list[DownloadedFile]:
    docs_dir = raw_dir / "hc3" / "docs"
    downloaded: list[DownloadedFile] = []
    for filename in PUBLIC_HC3_FILES.values():
        url = urljoin(HC3_FILE_BASE, filename)
        destination = docs_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            downloaded.append(DownloadedFile(destination.name, url, destination, destination.stat().st_size))
            continue
        downloaded.append(download_file(url, destination))
    return downloaded


def hc3_session_url(session_archive_name: str) -> str:
    return urljoin(HC3_NERSC_BASE, session_archive_name)
