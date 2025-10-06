from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


DATA_DIR = Path("./data")
LOG_FILE = DATA_DIR / "detections.csv"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectionLogRow:
    timestamp_iso: str
    hour: int
    source: str  # realtime | image | video
    class_label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    latitude: Optional[float]
    longitude: Optional[float]
    media_name: Optional[str]


def append_detection_log(
    *,
    source: str,
    class_label: str,
    confidence: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    media_name: Optional[str] = None,
) -> None:
    """Append a detection record to a CSV file.

    Creates the file with headers on first write.
    """
    _ensure_data_dir()
    # Auto-generate sequential media_name like pothole1, pothole2, ... if not provided
    if media_name is None or str(media_name).strip() == "":
        # Determine next index based on existing rows in the log file
        _ensure_data_dir()
        next_idx = 1
        if LOG_FILE.exists():
            try:
                with open(LOG_FILE, mode="r", encoding="utf-8", newline="") as rf:
                    reader = csv.DictReader(rf)
                    count_rows = 0
                    for _ in reader:
                        count_rows += 1
                    next_idx = count_rows + 1
            except Exception:
                next_idx = 1
        media_name = f"pothole{next_idx}"

    now_utc = datetime.now(timezone.utc)
    row = DetectionLogRow(
        timestamp_iso=now_utc.isoformat(),
        hour=int(now_utc.hour),
        source=source,
        class_label=class_label,
        confidence=float(confidence),
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
        latitude=float(latitude) if latitude is not None else None,
        longitude=float(longitude) if longitude is not None else None,
        media_name=media_name,
    )

    file_exists = LOG_FILE.exists()
    # If file exists but missing new columns (e.g., 'hour'), upgrade header by rewriting
    if file_exists:
        try:
            df_existing = pd.read_csv(LOG_FILE)
            target_cols = [
                "timestamp_iso",
                "hour",
                "source",
                "class_label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "latitude",
                "longitude",
                "media_name",
            ]
            changed = False
            for c in target_cols:
                if c not in df_existing.columns:
                    df_existing[c] = None
                    changed = True
            # Reorder columns if changed
            if changed:
                df_existing = df_existing[target_cols]
                df_existing.to_csv(LOG_FILE, index=False, encoding="utf-8")
        except Exception:
            pass
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_iso",
                "hour",
                "source",
                "class_label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "latitude",
                "longitude",
                "media_name",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(row))


