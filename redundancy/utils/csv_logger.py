"""Append-only CSV logging of per-epoch metrics."""

from __future__ import annotations

import csv
import os
from typing import Iterable, Mapping


class CSVLogger:
    """Write dict rows to a CSV file, creating parent directories as needed.

    The header is written from the first row's keys; subsequent rows must share
    those keys.
    """

    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Mapping[str, object]) -> None:
        with open(self.path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(dict(row))
