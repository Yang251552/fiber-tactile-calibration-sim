"""Config loading + a content hash for reproducibility."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    sha: str = field(init=False)

    def __post_init__(self) -> None:
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()[:12]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text())
    return Config(raw=raw, path=p)
