from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import requests


def download_and_extract_roboflow_zip(dataset_url: str, dest_dir: Path | str) -> Path:
	"""Download a Roboflow dataset ZIP and extract to dest_dir.

	Args:
		dataset_url: Public Roboflow dataset download URL (ends with ?key=...).
		dest_dir: Destination directory to extract the dataset.

	Returns:
		Destination directory as Path.
	"""
	dest_path = Path(dest_dir)
	dest_path.mkdir(parents=True, exist_ok=True)

	resp = requests.get(dataset_url, timeout=120)
	resp.raise_for_status()

	with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
		zf.extractall(dest_path)

	return dest_path


