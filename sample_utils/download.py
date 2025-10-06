import os
from pathlib import Path
from typing import Optional

import requests


def download_file(url: str, destination: Path | str, expected_size: Optional[int] = None) -> Path:
	"""Download a file from a URL to a destination path if missing or size mismatch.

	Args:
		url: Remote file URL.
		destination: Local path to save to. Parent directories will be created.
		expected_size: Optional expected file size in bytes to verify integrity.

	Returns:
		The destination path as Path.
	"""
	dest_path = Path(destination)
	dest_path.parent.mkdir(parents=True, exist_ok=True)

	# If exists and size matches (when provided), skip
	if dest_path.exists():
		if expected_size is None:
			return dest_path
		try:
			actual_size = dest_path.stat().st_size
			if actual_size == expected_size:
				return dest_path
		except OSError:
			pass

	with requests.get(url, stream=True, timeout=60) as response:
		response.raise_for_status()
		temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
		with open(temp_path, "wb") as f:
			for chunk in response.iter_content(chunk_size=1024 * 1024):
				if chunk:
					f.write(chunk)

	# Verify size if provided
	if expected_size is not None:
		actual_size = temp_path.stat().st_size
		if actual_size != expected_size:
			temp_path.unlink(missing_ok=True)
			raise ValueError(
				f"Downloaded size mismatch for {url}. Expected {expected_size}, got {actual_size}."
			)

	temp_path.replace(dest_path)
	return dest_path


