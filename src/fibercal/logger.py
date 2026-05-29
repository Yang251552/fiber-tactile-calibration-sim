"""Dataset I/O. The Logger writes Parquet; the ML side reads only Parquet and
never touches the sim or sensor model. The dataset file IS the deliverable
(work package: 'curation of dataset')."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dataset(rows: list[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def channel_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ch_")]
