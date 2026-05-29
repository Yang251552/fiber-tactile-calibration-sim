"""Fast pipeline tests (no 112k-row dataset; a few hundred rows from live presses)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fibercal import metrics, ml
from fibercal.hal import SimRig
from fibercal.schema import PHASE_HOLD, channel_columns, make_row


def test_config_loads_with_stable_sha(cfg):
    assert isinstance(cfg.sha, str) and len(cfg.sha) > 0
    assert cfg.seed == 42


def test_press_yields_60_steps_of_64_channels(cfg):
    rig = SimRig(cfg)
    steps = list(rig.press(20.0, 20.0, 5.0))
    assert len(steps) == 60  # ramp 25 + hold 10 + unload 25
    for state, ch in steps:
        assert len(ch) == 64


def test_hold_response_monotonic_in_force(cfg):
    rig = SimRig(cfg)

    def hold_peak(force):
        peak = 0.0
        for st, ch in rig.press(20.0, 20.0, force):
            if st.phase == PHASE_HOLD:
                peak = float(ch.max() - ch.min())
        return peak

    r1, r4, r8 = hold_peak(1.0), hold_peak(4.0), hold_peak(8.0)
    assert r1 < r4 < r8


def test_channel_columns():
    cols = channel_columns(64)
    assert cols[0] == "ch_00" and cols[-1] == "ch_63" and len(cols) == 64


def _tiny_dataset(cfg, seed=0):
    """A few hundred rows across locations/forces, incl. no-contact."""
    rig = SimRig(cfg)
    rows, sid, pid = [], 0, 0
    locs = [(10, 10), (20, 20), (30, 30), (15, 28), (28, 12)]
    forces = [0.0, 1.0, 3.0, 6.0, 9.0]  # 0.0 -> no-contact class
    for (x, y) in locs:
        for f in forces:
            for st, ch in rig.press(x, y, f):
                rows.append(make_row(st, ch, sample_id=sid, press_id=pid,
                                     sweep_type="grid", seed=seed,
                                     config_sha=cfg.sha, ch_cols=rig.channel_cols))
                sid += 1
            pid += 1
    return pd.DataFrame(rows)


def test_end_to_end_sensitivity_and_ml(cfg, size_mm):
    df = _tiny_dataset(cfg)
    assert len(df) > 300

    sens = metrics.sensitivity(df)
    assert sens["slope_per_N"] > 0

    tr, te = ml.random_split(df, seed=0)
    # press-grouped split: no press_id shared between train and test
    assert set(df.iloc[tr]["press_id"]).isdisjoint(set(df.iloc[te]["press_id"]))

    res = ml.evaluate(df, "linear", tr, te, seed=0)
    assert np.isfinite(res["force_r2"])
    assert np.isfinite(res["loc_err_median_mm"])
