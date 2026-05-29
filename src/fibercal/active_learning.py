"""Active-learning automated calibration.

The rig decides WHERE to probe next to calibrate with the fewest samples,
instead of sweeping a fixed grid. This directly matches the project's
"automated calibration / high-throughput" goal.

Acquisition: core-set / greedy k-center coverage in the sensor's feature space
(Sener & Savarese, 2018) -- each step adds the probe farthest from all already-
selected probes, so the labelled set covers the response manifold with minimal
redundancy.

Honesty note: on a *uniform* candidate pool, random sampling already covers the
space well and active learning shows little to no gain (verified). The benefit
appears with a *realistic biased pool* -- a real actuator revisits a few regions
and dwells in the saturated high-force regime -- where random inherits the bias
but core-set de-biases by spreading coverage. The headline result is a learning
curve: test force-RMSE vs number of samples, active vs random selection, on a
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
    """Steady-state (hold-phase) channel vector for one probe."""
    last = None
    for st, ch in rig.press(x, y, force):
        if st.phase == PHASE_HOLD:
            last = ch
    return last


def _candidate_pool(cfg: Config, rng, n, biased=False):
    """Pool of (x, y, force) probes. A real actuator rarely offers a perfectly
    uniform pool: it preferentially revisits a few regions and tends to dwell in
    the high-force (saturated) regime. With `biased=True` the pool is clustered
    in space and skewed in force -- the realistic case where coverage-based
    active selection beats random (which just inherits the bias)."""
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


def _collect(rig, probes):
    X = np.array([_hold_reading(rig, x, y, f) for (x, y, f) in probes])
    y = probes[:, 2]
    return X, y


def run_comparison(cfg: Config, seed=40, n_seed=15, n_total=150,
                   pool_size=600, batch=15):
    """Active vs random selection. Returns dict with two learning curves."""
    rng = np.random.default_rng(seed)
    rig = SimRig(cfg, rng=np.random.default_rng(seed))

    # fixed test set + candidate pool
    test_probes = _candidate_pool(cfg, np.random.default_rng(seed + 1), 250)
    Xte, yte = _collect(rig, test_probes)

    def rmse(model):
        return float(np.sqrt(np.mean((model.predict(Xte) - yte) ** 2)))

    def loop(strategy):
        pool = _candidate_pool(cfg, np.random.default_rng(seed + 2), pool_size,
                               biased=True)
        Xpool, ypool = _collect(rig, pool)
        chosen = list(rng.choice(len(pool), n_seed, replace=False))
        curve = []
        while True:
            X, y = Xpool[chosen], ypool[chosen]
            ens = _Ensemble(seed=seed)
            ens.fit(X, y)
            curve.append((len(chosen), rmse(ens)))
            if len(chosen) >= n_total:
                break
            remaining = [i for i in range(len(pool)) if i not in set(chosen)]
            if strategy == "random":
                pick = list(rng.choice(remaining, batch, replace=False))
            else:  # active: core-set / greedy k-center coverage in feature space
                rem = np.array(remaining)
                Xr = ens.scaler.transform(Xpool[rem])
                Xc = ens.scaler.transform(Xpool[chosen])
                # distance of each remaining point to the nearest selected point
                mind = np.min(np.linalg.norm(Xr[:, None, :] - Xc[None, :, :], axis=2),
                              axis=1)
                pick = []
                for _ in range(batch):
                    j = int(np.argmax(mind))
                    pick.append(int(rem[j]))
                    # update min-distances with the newly added centre
                    dnew = np.linalg.norm(Xr - Xr[j], axis=1)
                    mind = np.minimum(mind, dnew)
                    mind[j] = -1.0  # don't pick again
            chosen.extend(pick)
        return np.array(curve)

    active = loop("active")
    random = loop("random")
    return {"active": active, "random": random}


def sample_efficiency(curves: dict) -> dict:
    """How many fewer samples active needs to hit random's final RMSE."""
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
