#!/usr/bin/env python3
"""Active-learning automated calibration: active vs random sample selection.

    python scripts/run_active_learning.py

Saves figures/active_learning.png and prints the sample-efficiency gain.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fibercal.active_learning import run_comparison, sample_efficiency
from fibercal.config import load_config


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config()
    print("[active-learning] running active vs random calibration...")
    curves = run_comparison(cfg)
    eff = sample_efficiency(curves)

    print(f"  random final RMSE  : {curves['random'][-1, 1]:.3f} N "
          f"@ {eff['n_random']} samples")
    print(f"  active reaches it  : @ {eff['n_active']} samples "
          f"({eff['sample_saving_pct']:.0f}% fewer)")
    print(f"  active final RMSE  : {eff['active_final_rmse']:.3f} N")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(curves["random"][:, 0], curves["random"][:, 1], "o-", label="random")
    ax.plot(curves["active"][:, 0], curves["active"][:, 1], "s-", label="active (uncertainty)")
    ax.axhline(eff["target_rmse"], color="gray", ls="--", lw=1,
               label="random final RMSE")
    ax.set_xlabel("# calibration samples")
    ax.set_ylabel("test force RMSE (N)")
    ax.set_title(f"Automated calibration: active learning saves "
                 f"{eff['sample_saving_pct']:.0f}% samples")
    ax.legend()
    out = root / "figures" / "active_learning.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
