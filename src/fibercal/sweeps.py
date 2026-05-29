"""High-throughput automated data collection.

Four sweep types mirror a real calibration campaign:
  grid    - deterministic location x force grid (clean calibration curves)
  random  - domain-randomized pool incl. sub-threshold no-contact (ML training)
  ramp    - fine loading/unloading ramps at fixed points (hysteresis)
  repeat  - same target probed K times (repeatability)

Each press emits ~60 trajectory rows sharing one press_id, so downstream splits
can group on press_id and avoid leaking near-duplicate rows.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .hal import TactileRig
from .schema import make_row


def _emit_press(rig: TactileRig, x, y, force, *, sweep_type, seed, sha,
                sample_counter, press_counter, repeat_id=-1, rows=None):
    press_id = press_counter[0]
    for state, channels in rig.press(x, y, force):
        rows.append(make_row(
            state, channels,
            sample_id=sample_counter[0], press_id=press_id, sweep_type=sweep_type,
            seed=seed, config_sha=sha, repeat_id=repeat_id,
            ch_cols=rig.channel_cols,
        ))
        sample_counter[0] += 1
    press_counter[0] += 1


def generate_dataset(rig: TactileRig, cfg: Config) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    sha = cfg.sha
    seed = cfg.seed
    size = float(cfg["surface"]["size_mm"])
    margin = float(cfg["surface"]["margin_mm"])
    lo, hi = margin, size - margin
    rows: list[dict] = []
    counter = [0]
    presses = [0]

    # --- grid sweep ---
    g = cfg["sweeps"]["grid"]
    locs = np.linspace(lo, hi, int(g["locations_per_side"]))
    for gx in locs:
        for gy in locs:
            for f in g["force_levels"]:
                _emit_press(rig, gx, gy, f, sweep_type="grid",
                            seed=seed, sha=sha, sample_counter=counter,
                            press_counter=presses, rows=rows)

    # --- random sweep (domain randomization, includes no-contact) ---
    r = cfg["sweeps"]["random"]
    for _ in range(int(r["n_samples"])):
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        f = rng.uniform(float(r["force_min"]), float(r["force_max"]))
        _emit_press(rig, x, y, f, sweep_type="random",
                    seed=seed, sha=sha, sample_counter=counter,
                    press_counter=presses, rows=rows)

    # --- ramp sweep (hysteresis) ---
    rp = cfg["sweeps"]["ramp"]
    for (x, y) in rp["locations"]:
        for f in np.linspace(0.0, float(rp["force_max"]), int(rp["n_levels"])):
            _emit_press(rig, x, y, float(f), sweep_type="ramp",
                        seed=seed, sha=sha, sample_counter=counter,
                        press_counter=presses, rows=rows)

    # --- repeat sweep (repeatability) ---
    rep = cfg["sweeps"]["repeat"]
    for rid, (x, y, f) in enumerate(rep["targets"]):
        for _ in range(int(rep["k"])):
            _emit_press(rig, x, y, float(f), sweep_type="repeat",
                        seed=seed, sha=sha, sample_counter=counter,
                        press_counter=presses, repeat_id=rid, rows=rows)

    return rows
