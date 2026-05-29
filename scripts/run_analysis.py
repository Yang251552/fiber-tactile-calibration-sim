#!/usr/bin/env python3
"""Train inverse models, characterize the sensor, and emit figures.

    python scripts/run_analysis.py [--data data/dataset.parquet]
                                   [--figdir figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibercal import metrics, ml, viz
from fibercal.config import load_config
from fibercal.hal import SimRig
from fibercal.logger import load_dataset


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/inf) with None so the result
    is strictly valid JSON (json allows NaN by default; strict parsers reject it)."""
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.parquet")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = load_dataset(root / args.data)
    cfg = load_config(args.config)
    size = float(cfg["surface"]["size_mm"])
    figdir = root / args.figdir
    print(f"[data] {len(df):,} rows")

    # --- characterization ---
    rig = SimRig(cfg)
    char = metrics.characterize(df, rig=rig)
    print("\n=== SENSOR CHARACTERIZATION ===")
    print(f"  sensitivity        : {char['sensitivity']['slope_per_N']:.3f} /N "
          f"(linearity R²={char['sensitivity']['linearity_r2']:.3f})")
    print(f"  force resolution   : {char['resolution']['min_detectable_dF_N']*1000:.1f} mN")
    print(f"  detection floor    : {char['range']['detection_floor_N']:.2f} N")
    print(f"  saturation onset   : {char['range']['saturation_onset_N']:.1f} N")
    hyst = char["hysteresis"]
    hyst_ok = hyst.get("available", True)
    print(f"  hysteresis         : "
          + (f"{hyst['hysteresis_pct_fs']:.1f} % FS" if hyst_ok
             else "n/a (no unload phase in dataset)"))
    print(f"  repeatability σ_F  : {char['repeatability']['force_std_N']*1000:.1f} mN")

    # --- inverse models ---
    print("\n=== INVERSE MODELS ===")
    results = ml.run_all(df, size, cfg.seed)
    summary = {}
    for (split, kind), r in results.items():
        key = f"{split}/{kind}"
        summary[key] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"  {key:24s}  det_F1={r['detection_f1']:.3f}  "
              f"force_R²={r['force_r2']:.3f}  MAE={r['force_mae_N']*1000:.0f}mN  "
              f"loc_med={r['loc_err_median_mm']:.2f}mm")

    # --- figures ---
    print("\n=== FIGURES ===")
    for f in (
        viz.spatial_heatmap(rig, 20.0, 20.0, 6.0, figdir),
        viz.datasheet(char, figdir),
        viz.ml_results(results, figdir),
    ):
        print(f"  wrote {f}")

    # --- machine-readable results ---
    res_path = figdir / "results.json"
    char_summary = {
        "sensitivity_per_N": char["sensitivity"]["slope_per_N"],
        "linearity_r2": char["sensitivity"]["linearity_r2"],
        "resolution_mN": char["resolution"]["min_detectable_dF_N"] * 1000,
        "detection_floor_N": char["range"]["detection_floor_N"],
        "saturation_onset_N": char["range"]["saturation_onset_N"],
        "hysteresis_available": bool(hyst_ok),
        # null (not NaN) when unavailable -> strictly valid JSON
        "hysteresis_pct_fs": hyst["hysteresis_pct_fs"] if hyst_ok else None,
        "repeatability_force_std_mN": char["repeatability"]["force_std_N"] * 1000,
    }
    payload = _json_safe({"characterization": char_summary, "models": summary,
                          "config_sha": cfg.sha, "seed": cfg.seed})
    # allow_nan=False guarantees the file is parseable by strict JSON readers.
    res_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"  wrote {res_path}")


if __name__ == "__main__":
    main()
