"""Sensor characterization (work package 2: resolution / sensitivity / range).

Metrics are computed from the dataset's ground-truth force and the raw sensor
response, plus a dedicated drift probe that drives the sensor model directly.
Each function returns both headline numbers and arrays for plotting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .logger import channel_cols
from .schema import PHASE_HOLD, PHASE_RAMP, PHASE_UNLOAD


def peak_response(df: pd.DataFrame) -> np.ndarray:
    """Per-row peak channel reading above the array's resting minimum."""
    cols = channel_cols(df)
    arr = df[cols].values
    return arr.max(axis=1) - arr.min(axis=1)


def sensitivity(df: pd.DataFrame) -> dict:
    """Slope of peak response vs applied force on steady-state grid samples."""
    sub = df[(df["sweep_type"] == "grid") & (df["phase"] == PHASE_HOLD)].copy()
    f = sub["applied_force_N"].values
    r = peak_response(sub)
    A = np.vstack([f, np.ones_like(f)]).T
    slope, intercept = np.linalg.lstsq(A, r, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((r - pred) ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope_per_N": float(slope), "intercept": float(intercept),
            "linearity_r2": float(r2), "force": f, "response": r}


def repeatability(df: pd.DataFrame, sens_slope: float) -> dict:
    """Std of response (and equiv. force) across repeats at fixed targets."""
    sub = df[(df["sweep_type"] == "repeat") & (df["phase"] == PHASE_HOLD)].copy()
    sub = sub.assign(resp=peak_response(sub))
    per = sub.groupby("repeat_id")["resp"].agg(["mean", "std", "count"])
    resp_std = float(per["std"].mean())
    force_std = resp_std / sens_slope if sens_slope > 0 else float("nan")
    return {"response_std": resp_std, "force_std_N": force_std, "per_target": per}


def resolution(df: pd.DataFrame, sens_slope: float, k: float = 2.0) -> dict:
    """Minimum distinguishable force step ~ k * response noise / sensitivity."""
    rep = repeatability(df, sens_slope)
    min_dF = k * rep["response_std"] / sens_slope if sens_slope > 0 else float("nan")
    return {"min_detectable_dF_N": float(min_dF), "k": k,
            "response_std": rep["response_std"]}


def force_range(df: pd.DataFrame, sens: dict, noise_k: float = 3.0) -> dict:
    """Detection floor (force where response clears noise) and saturation onset."""
    sub = df[(df["sweep_type"] == "grid") & (df["phase"] == PHASE_HOLD)].copy()
    sub = sub.assign(resp=peak_response(sub))
    by_f = sub.groupby("target_force_N")["resp"].mean()
    forces = by_f.index.values
    resp = by_f.values

    # detection floor: response above baseline noise band
    rep = repeatability(df, sens["slope_per_N"])
    floor_resp = noise_k * rep["response_std"]
    above = forces[resp > floor_resp]
    f_min = float(above.min()) if above.size else float(forces.max())

    # saturation onset: where local slope drops below 50% of small-force slope
    dr = np.gradient(resp, forces)
    ref = dr[: max(1, len(dr) // 3)].mean()
    sat_idx = np.where(dr < 0.5 * ref)[0]
    f_sat = float(forces[sat_idx[0]]) if sat_idx.size else float(forces.max())

    return {"detection_floor_N": f_min, "saturation_onset_N": f_sat,
            "max_tested_N": float(forces.max()), "forces": forces, "response": resp}


def hysteresis(df: pd.DataFrame) -> dict:
    """Loading vs unloading response gap from ramp sweeps -> % of full scale."""
    sub = df[df["sweep_type"] == "ramp"].copy()
    sub = sub.assign(resp=peak_response(sub))
    load = sub[sub["phase"] == PHASE_RAMP]
    unload = sub[sub["phase"] == PHASE_UNLOAD]
    bins = np.linspace(0, sub["applied_force_N"].max(), 16)
    centers = 0.5 * (bins[:-1] + bins[1:])
    lo = load.groupby(pd.cut(load["applied_force_N"], bins),
                      observed=False)["resp"].mean().values
    un = unload.groupby(pd.cut(unload["applied_force_N"], bins),
                        observed=False)["resp"].mean().values
    full_scale = np.nanmax(sub["resp"].values)
    # only compare bins where both phases have data
    valid = ~(np.isnan(lo) | np.isnan(un))
    gap = float(np.max(np.abs(un[valid] - lo[valid]))) if valid.any() else 0.0
    pct = 100.0 * gap / full_scale if full_scale > 0 else 0.0
    return {"max_gap": float(gap), "hysteresis_pct_fs": float(pct),
            "force_bins": centers, "loading": lo, "unloading": un}


def drift_probe(rig, x: float, y: float, force: float, n_steps: int = 400) -> dict:
    """Drive the sensor model directly (no reset) to expose slow drift."""
    from .contact import make_state
    from .schema import PHASE_HOLD as HOLD
    rig.sensor.reset()
    series = []
    for i in range(n_steps):
        st = make_state(t=i * 0.02, target_force_N=force, applied_force_N=force,
                        contact_x=x, contact_y=y, phase=HOLD,
                        indenter_radius_mm=rig.radius)
        ch = rig.sensor.read(st)
        series.append(ch.max() - ch.min())
    series = np.array(series)
    return {"series": series, "drift_total": float(series[-50:].mean() - series[:50].mean())}


def characterize(df: pd.DataFrame, rig=None) -> dict:
    sens = sensitivity(df)
    rep = repeatability(df, sens["slope_per_N"])
    res = resolution(df, sens["slope_per_N"])
    rng = force_range(df, sens)
    hys = hysteresis(df)
    out = {"sensitivity": sens, "repeatability": rep, "resolution": res,
           "range": rng, "hysteresis": hys}
    if rig is not None:
        out["drift"] = drift_probe(rig, 20.0, 20.0, 5.0)
    return out
