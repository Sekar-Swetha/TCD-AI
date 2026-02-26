# src/analytics/logger.py
from __future__ import annotations
import csv
import os
from typing import Dict, Iterable, List

from .metrics import RunMetrics


class CSVLogger:
    """
    Appends RunMetrics rows into a CSV file.
    Creates the file with header if it doesn't exist.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def _ensure_header(self, fieldnames: List[str]) -> None:
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, metrics: RunMetrics) -> None:
        row = metrics.to_row()
        fieldnames = list(row.keys())
        self._ensure_header(fieldnames)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def log_many(self, metrics_list: Iterable[RunMetrics]) -> None:
        metrics_list = list(metrics_list)
        if not metrics_list:
            return
        first = metrics_list[0].to_row()
        fieldnames = list(first.keys())
        self._ensure_header(fieldnames)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for m in metrics_list:
                writer.writerow(m.to_row())