"""The dataset contract. Defined once; the sim, the logger and the ML side
all import column names from here so they can never drift apart."""
from __future__ import annotations

from dataclasses import asdict, dataclass


def channel_columns(n_channels: int) -> list[str]:
    """Column names for the sensor channels: ch_00, ch_01, ..."""
    width = max(2, len(str(n_channels - 1)))
    return [f"ch_{i:0{width}d}" for i in range(n_channels)]


# Phases of one force-application "press".
PHASE_APPROACH = "approach"
PHASE_RAMP = "ramp"       # loading
PHASE_HOLD = "hold"
PHASE_UNLOAD = "unload"   # unloading
PHASE_RETRACT = "retract"


@dataclass
class ContactState:
    """Ground-truth contact, independent of any physics engine.
    The sensor model depends ONLY on this -> engine is swappable."""
    t: float
    target_force_N: float
    applied_force_N: float
    contact_x: float          # mm, surface coords
    contact_y: float          # mm
    in_contact: bool
    phase: str
    indenter_radius_mm: float


# Non-channel, non-meta label/feature columns of a dataset row.
LABEL_COLUMNS = [
    "sample_id",
    "press_id",
    "sweep_type",
    "seed",
    "config_sha",
    "repeat_id",
    "t",
    "phase",
    "target_force_N",
    "applied_force_N",
    "contact_x",
    "contact_y",
    "in_contact",
    "indenter_radius_mm",
]


def make_row(
    state: ContactState,
    channels,
    *,
    sample_id: int,
    press_id: int,
    sweep_type: str,
    seed: int,
    config_sha: str,
    repeat_id: int = -1,
    ch_cols: list[str] | None = None,
) -> dict:
    """Assemble one dataset row from ground truth + channel readings.

    press_id groups all trajectory rows of a single press; splits MUST group on
    it to avoid leaking near-duplicate rows across train/test."""
    if ch_cols is None:
        ch_cols = channel_columns(len(channels))
    row = {
        "sample_id": sample_id,
        "press_id": press_id,
        "sweep_type": sweep_type,
        "seed": seed,
        "config_sha": config_sha,
        "repeat_id": repeat_id,
        "t": state.t,
        "phase": state.phase,
        "target_force_N": state.target_force_N,
        "applied_force_N": state.applied_force_N,
        "contact_x": state.contact_x,
        "contact_y": state.contact_y,
        "in_contact": state.in_contact,
        "indenter_radius_mm": state.indenter_radius_mm,
    }
    row.update({c: float(v) for c, v in zip(ch_cols, channels)})
    return row
