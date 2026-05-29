"""Inverse / calibration models: channel readings -> (contact?, location, force).

Model ladder: linear/logistic baseline (the one to beat) -> MLP. Two splits are
reported: a press-grouped random split (headline accuracy, no leakage) and a
spatial-holdout split (held-out surface region -> honest generalization to
unprobed locations)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .logger import channel_cols


@dataclass
class SplitResult:
    name: str
    metrics: dict = field(default_factory=dict)


def random_split(df: pd.DataFrame, seed: int, frac=0.2):
    """Group-aware split: all ~60 trajectory rows of one press stay together.

    Splitting at the row level would scatter near-duplicate rows of a single
    press across train and test, leaking information and inflating scores. We
    split on press_id so train and test never share a press."""
    if "press_id" not in df.columns:
        raise KeyError("dataset lacks press_id; regenerate with current sweeps.py")
    rng = np.random.default_rng(seed)
    presses = df["press_id"].unique()
    rng.shuffle(presses)
    n_test = int(len(presses) * frac)
    test_presses = set(presses[:n_test].tolist())
    is_test = df["press_id"].isin(test_presses).values
    return np.where(~is_test)[0], np.where(is_test)[0]


def spatial_holdout_split(df: pd.DataFrame, size_mm: float):
    """Hold out one surface quadrant entirely from training."""
    test_mask = (df["contact_x"] > size_mm / 2) & (df["contact_y"] > size_mm / 2)
    train_idx = np.where(~test_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]
    return train_idx, test_idx


def _make_models(kind: str, seed: int):
    if kind == "linear":
        det = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        frc = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        loc = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:  # mlp
        det = make_pipeline(StandardScaler(),
                            MLPClassifier((128, 64), max_iter=400, random_state=seed))
        frc = make_pipeline(StandardScaler(),
                            MLPRegressor((128, 64), max_iter=400, random_state=seed))
        loc = make_pipeline(StandardScaler(),
                            MLPRegressor((128, 64), max_iter=400, random_state=seed))
    return det, frc, loc


def evaluate(df: pd.DataFrame, kind: str, train_idx, test_idx, seed: int) -> dict:
    cols = channel_cols(df)
    X = df[cols].values
    y_det = df["in_contact"].astype(int).values
    y_force = df["applied_force_N"].values
    y_loc = df[["contact_x", "contact_y"]].values

    det, frc, loc = _make_models(kind, seed)

    # Detection (all samples).
    det.fit(X[train_idx], y_det[train_idx])
    det_pred = det.predict(X[test_idx])
    f1 = f1_score(y_det[test_idx], det_pred)

    # Force + localization: train/eval on in-contact samples only.
    tr_c = train_idx[y_det[train_idx] == 1]
    te_c = test_idx[y_det[test_idx] == 1]

    frc.fit(X[tr_c], y_force[tr_c])
    f_pred = frc.predict(X[te_c])
    force_mae = mean_absolute_error(y_force[te_c], f_pred)
    force_rmse = float(np.sqrt(mean_squared_error(y_force[te_c], f_pred)))
    force_r2 = r2_score(y_force[te_c], f_pred)

    loc.fit(X[tr_c], y_loc[tr_c])
    l_pred = loc.predict(X[te_c])
    loc_err = np.linalg.norm(l_pred - y_loc[te_c], axis=1)

    return {
        "model": kind,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "detection_f1": float(f1),
        "force_mae_N": float(force_mae),
        "force_rmse_N": float(force_rmse),
        "force_r2": float(force_r2),
        "loc_err_median_mm": float(np.median(loc_err)),
        "loc_err_p90_mm": float(np.percentile(loc_err, 90)),
        # kept for plotting
        "_force_true": y_force[te_c],
        "_force_pred": f_pred,
        "_loc_true": y_loc[te_c],
        "_loc_pred": l_pred,
    }


def run_all(df: pd.DataFrame, size_mm: float, seed: int) -> dict:
    """Train every model on every split; return a nested result dict."""
    out = {}
    splits = {
        "random": random_split(df, seed),
        "spatial_holdout": spatial_holdout_split(df, size_mm),
    }
    for split_name, (tr, te) in splits.items():
        for kind in ("linear", "mlp"):
            out[(split_name, kind)] = evaluate(df, kind, tr, te, seed)
    return out
