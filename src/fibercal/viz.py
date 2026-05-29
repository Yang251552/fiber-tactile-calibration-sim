"""Figures for the README / report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def spatial_heatmap(rig, x, y, force, outdir):
    """8x8 channel heatmap for one contact -> shows load spreading."""
    from .contact import make_state
    from .schema import PHASE_HOLD
    rig.sensor.reset()
    st = make_state(t=0, target_force_N=force, applied_force_N=force,
                    contact_x=x, contact_y=y, phase=PHASE_HOLD,
                    indenter_radius_mm=rig.radius)
    ch = rig.sensor.read(st)
    n = rig.sensor.n_side
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(ch.reshape(n, n), origin="lower", cmap="inferno",
                   extent=[0, rig.cfg["surface"]["size_mm"]] * 2)
    ax.plot(x, y, "c+", markersize=14, markeredgewidth=2, label="true contact")
    ax.set_title(f"8x8 fiber response  (contact @ {force:.0f} N)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="reading")
    return _save(fig, Path(outdir) / "spatial_heatmap.png")


def datasheet(char: dict, outdir):
    """One-page auto-generated sensor 'datasheet'."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))

    # sensitivity
    s = char["sensitivity"]
    ax = axs[0, 0]
    ax.scatter(s["force"], s["response"], s=8, alpha=0.4)
    xs = np.linspace(0, s["force"].max(), 50)
    ax.plot(xs, s["slope_per_N"] * xs + s["intercept"], "r-")
    ax.set_title(f"Sensitivity = {s['slope_per_N']:.3f} /N  (R²={s['linearity_r2']:.3f})")
    ax.set_xlabel("applied force (N)"); ax.set_ylabel("peak response")

    # range
    r = char["range"]
    ax = axs[0, 1]
    ax.plot(r["forces"], r["response"], "o-")
    ax.axvline(r["detection_floor_N"], color="g", ls="--",
               label=f"floor {r['detection_floor_N']:.2f} N")
    ax.axvline(r["saturation_onset_N"], color="m", ls="--",
               label=f"sat {r['saturation_onset_N']:.1f} N")
    ax.set_title("Dynamic range"); ax.set_xlabel("force (N)")
    ax.set_ylabel("response"); ax.legend(fontsize=8)

    # hysteresis
    h = char["hysteresis"]
    ax = axs[0, 2]
    ax.plot(h["force_bins"], h["loading"], "b.-", label="loading")
    ax.plot(h["force_bins"], h["unloading"], "r.-", label="unloading")
    ax.set_title(f"Hysteresis = {h['hysteresis_pct_fs']:.1f} % FS")
    ax.set_xlabel("force (N)"); ax.set_ylabel("response"); ax.legend(fontsize=8)

    # repeatability
    rep = char["repeatability"]
    ax = axs[1, 0]
    per = rep["per_target"]
    ax.bar(range(len(per)), per["mean"], yerr=per["std"] * 20, capsize=4)
    ax.set_title(f"Repeatability  σ_F ≈ {rep['force_std_N']*1000:.1f} mN\n(error bars ×20)")
    ax.set_xlabel("repeat target"); ax.set_ylabel("mean response")

    # resolution
    res = char["resolution"]
    ax = axs[1, 1]
    ax.axis("off")
    txt = (f"Resolution\n\nmin detectable ΔF\n≈ {res['min_detectable_dF_N']*1000:.1f} mN\n"
           f"(k={res['k']:.0f}·σ rule)\n\nresponse noise σ\n= {res['response_std']:.4f}")
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13,
            transform=ax.transAxes)
    ax.set_title("Force resolution")

    # drift
    ax = axs[1, 2]
    if "drift" in char:
        d = char["drift"]
        ax.plot(d["series"])
        ax.set_title(f"Drift over hold\nΔ ≈ {d['drift_total']:.4f}")
        ax.set_xlabel("step"); ax.set_ylabel("response")
    else:
        ax.axis("off")

    fig.suptitle("Simulated optical-fiber tactile sensor — characterization datasheet",
                 fontsize=14)
    return _save(fig, Path(outdir) / "datasheet.png")


def ml_results(results: dict, outdir):
    """Predicted-vs-true force + localization scatter for the MLP/random split."""
    res = results[("random", "mlp")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))

    axs[0].scatter(res["_force_true"], res["_force_pred"], s=6, alpha=0.3)
    lim = [0, max(res["_force_true"].max(), res["_force_pred"].max())]
    axs[0].plot(lim, lim, "k--")
    axs[0].set_title(f"Force: R²={res['force_r2']:.3f}, "
                     f"MAE={res['force_mae_N']*1000:.0f} mN")
    axs[0].set_xlabel("true force (N)"); axs[0].set_ylabel("predicted (N)")

    lt, lp = res["_loc_true"], res["_loc_pred"]
    axs[1].quiver(lt[:, 0], lt[:, 1], lp[:, 0] - lt[:, 0], lp[:, 1] - lt[:, 1],
                  angles="xy", scale_units="xy", scale=1, width=0.003, alpha=0.5)
    axs[1].set_title(f"Localization: median err "
                     f"{res['loc_err_median_mm']:.2f} mm")
    axs[1].set_xlabel("x (mm)"); axs[1].set_ylabel("y (mm)")
    return _save(fig, Path(outdir) / "ml_results.png")
