"""Active-learning automated calibration.

The rig decides WHICH probes to actually execute so it can calibrate with the
fewest physical presses, instead of sweeping a fixed grid. This matches the
project's "automated calibration / high-throughput" goal.

Acquisition: core-set / greedy k-center coverage (Sener & Savarese, 2018) in the
**command space** (x, y, force) of the candidate probes. Crucially this uses
ONLY information available *before* probing -- the commanded coordinates -- so
the decision costs no presses. Only the selected probes are ever executed, so
the x-axis ("# calibration samples") equals the number of real presses, and the
"fewer samples" claim is hardware-valid (no hidden up-front probing of the pool).

Honesty note: on a *uniform* candidate pool, random sampling already covers the
space well and active learning shows little to no gain (verified). The benefit
appears with a *realistic biased pool* -- a real actuator revisits a few regions
and dwells in the saturated high-force regime -- where random inherits the bias
but core-set de-biases by spreading coverage. The headline result is a learning
curve: test force-RMSE vs number of executed presses, active vs random, on a
held-out UNIFORM test set.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .config import Config
from .hal import SimRig
from .schema import PHASE_HOLD


def _hold_reading(rig: SimRig, x, y, force):
    """Steady-state (hold-phase) channel vector for one probe (one press)."""
    last = None
    for st, ch in rig.press(x, y, force):
        if st.phase == PHASE_HOLD:
            last = ch
    return last


def _candidate_pool(cfg: Config, rng, n, biased=False):
    """Pool of candidate (x, y, force) probe COMMANDS (not yet executed).

    A real actuator rarely offers a perfectly uniform pool: it preferentially
    revisits a few regions and tends to dwell in the high-force (saturated)
    regime. With `biased=True` the pool is clustered in space and skewed in
    force -- the realistic case where coverage-based active selection beats
    random (which just inherits the bias)."""
    size = float(cfg["surface"]["size_mm"])
    m = float(cfg["surface"]["margin_mm"])
    fmax = float(cfg["sweeps"]["random"]["force_max"])
    if not biased:
        xy = rng.uniform(m, size - m, size=(n, 2))
        f = rng.uniform(0.2, fmax, size=n)
    else:
        # 3 spatial clusters + force skewed toward the saturated high end
        centers = rng.uniform(m + 5, size - m - 5, size=(3, 2))
        which = rng.integers(0, 3, size=n)
        xy = np.clip(centers[which] + rng.normal(0, 3.5, size=(n, 2)), m, size - m)
        f = 0.2 + (fmax - 0.2) * rng.beta(3.0, 1.2, size=n)  # mass near fmax
    return np.column_stack([xy, f])  # (n, 3): x, y, force


class _Ensemble:
    """Bagged MLP regressor (channels -> force)."""

    def __init__(self, n_members=4, seed=0):
        self.n = n_members
        self.seed = seed
        self.scaler = StandardScaler()
        self.members: list[MLPRegressor] = []

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for i in range(self.n):
            idx = rng.integers(0, len(X), len(X))  # bootstrap
            m = MLPRegressor((64, 32), max_iter=300, random_state=self.seed + i)
            m.fit(Xs[idx], y[idx])
            self.members.append(m)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return np.mean([m.predict(Xs) for m in self.members], axis=0)


def _coreset_pick(cmd_norm, chosen, remaining, batch):
    """Greedy k-center: add points farthest (in normalised command space) from
    the already-chosen set. Uses only command coordinates -> no probing."""
    rem = np.array(remaining)
    Xr = cmd_norm[rem]
    Xc = cmd_norm[chosen]
    mind = np.min(np.linalg.norm(Xr[:, None, :] - Xc[None, :, :], axis=2), axis=1)
    pick = []
    for _ in range(batch):
        j = int(np.argmax(mind))
        pick.append(int(rem[j]))
        mind = np.minimum(mind, np.linalg.norm(Xr - Xr[j], axis=1))
        mind[j] = -1.0
    return pick


def run_comparison(cfg: Config, seed=40, n_seed=15, n_total=150,
                   pool_size=600, batch=15):
    """Active vs random selection. Returns two learning curves of
    (n_presses_executed, test_force_RMSE)."""
    rng = np.random.default_rng(seed)
    rig = SimRig(cfg, rng=np.random.default_rng(seed))

    # Fixed held-out UNIFORM test set (simulated evaluation ground truth).
    test_probes = _candidate_pool(cfg, np.random.default_rng(seed + 1), 250)
    Xte = np.array([_hold_reading(rig, *p) for p in test_probes])
    yte = test_probes[:, 2]

    def rmse(model):
        return float(np.sqrt(np.mean((model.predict(Xte) - yte) ** 2)))

    def loop(strategy):
        pool = _candidate_pool(cfg, np.random.default_rng(seed + 2), pool_size,
                               biased=True)
        # normalised command-space coords (known before any probing)
        cmd_norm = (pool - pool.mean(0)) / (pool.std(0) + 1e-9)

        cache: dict[int, np.ndarray] = {}  # only EXECUTED probes get pressed

        def reading(i):
            if i not in cache:
                cache[i] = _hold_reading(rig, *pool[i])  # one real press
            return cache[i]

        chosen = list(rng.choice(len(pool), n_seed, replace=False))
        curve = []
        while True:
            X = np.array([reading(i) for i in chosen])
            y = pool[chosen, 2]
            ens = _Ensemble(seed=seed)
            ens.fit(X, y)
            # x-axis = presses actually executed (== len(chosen) == len(cache))
            curve.append((len(cache), rmse(ens)))
            if len(chosen) >= n_total:
                break
            remaining = [i for i in range(len(pool)) if i not in set(chosen)]
            if strategy == "random":
                pick = list(rng.choice(remaining, batch, replace=False))
            else:  # core-set coverage in command space (no probing to decide)
                pick = _coreset_pick(cmd_norm, chosen, remaining, batch)
            chosen.extend(pick)
        return np.array(curve)

    return {"active": loop("active"), "random": loop("random")}


def sample_efficiency(curves: dict) -> dict:
    """How many fewer presses active needs to hit random's final RMSE."""
    rnd = curves["random"]
    act = curves["active"]
    target = rnd[-1, 1]
    hit = act[act[:, 1] <= target]
    n_active = int(hit[0, 0]) if len(hit) else int(act[-1, 0])
    n_random = int(rnd[-1, 0])
    gain = 100.0 * (1 - n_active / n_random)
    return {"target_rmse": float(target), "n_active": n_active,
            "n_random": n_random, "sample_saving_pct": float(gain),
            "active_final_rmse": float(act[-1, 1])}
