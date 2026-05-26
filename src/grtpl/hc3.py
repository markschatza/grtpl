from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import numpy as np
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


def buzsaki_hc3_session_base_url(topdir: str, session: str) -> str:
    return f"https://buzsakilab.nyumc.org/datasets/MizusekiK/hc-3/{topdir}/{session}/"


def download_buzsaki_hc3_session_files(
    topdir: str,
    session: str,
    raw_dir: Path,
    extensions: tuple[str, ...] = ("xml", "eeg"),
) -> list[DownloadedFile]:
    session_dir = raw_dir / "hc3" / topdir / session
    base_url = buzsaki_hc3_session_base_url(topdir, session)
    downloaded: list[DownloadedFile] = []
    for extension in extensions:
        filename = f"{session}.{extension}"
        destination = session_dir / filename
        url = urljoin(base_url, filename)
        if destination.exists() and destination.stat().st_size > 0:
            downloaded.append(DownloadedFile(filename, url, destination, destination.stat().st_size))
            continue
        downloaded.append(download_file(url, destination))
    return downloaded


def read_hc3_xml_metadata(xml_path: Path) -> dict[str, object]:
    root = ET.parse(xml_path).getroot()
    acquisition = root.find("acquisitionSystem")
    field_potentials = root.find("fieldPotentials")
    if acquisition is None or field_potentials is None:
        raise ValueError(f"Missing acquisition metadata in {xml_path}")

    channel_groups: list[list[int]] = []
    groups_root = root.find("anatomicalDescription/channelGroups")
    if groups_root is not None:
        for group in groups_root.findall("group"):
            channel_groups.append([int(channel.text) for channel in group.findall("channel")])

    return {
        "n_channels": int(acquisition.findtext("nChannels")),
        "sampling_rate_hz": float(acquisition.findtext("samplingRate")),
        "lfp_sampling_rate_hz": float(field_potentials.findtext("lfpSamplingRate")),
        "channel_groups": channel_groups,
    }


def load_eeg_channels(eeg_path: Path, n_channels: int, channels: list[int]) -> np.ndarray:
    raw = np.memmap(eeg_path, dtype="<i2", mode="r")
    if raw.size % n_channels != 0:
        raise ValueError(f"{eeg_path} sample count is not divisible by {n_channels} channels")
    samples = raw.reshape((-1, n_channels))
    return np.asarray(samples[:, channels], dtype=np.float32)
