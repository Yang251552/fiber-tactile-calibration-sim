"""Virtual actuator + contact physics.

Ground truth is analytic (Hertzian indentation) rather than read from a noisy
physics engine, so the 'applied force' is clean and fully reproducible. The
closed-loop 'press to target force' controller here is exactly the logic that
would move to an Arduino + stepper + load cell on real hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import (
    PHASE_HOLD,
    PHASE_RAMP,
    PHASE_UNLOAD,
    ContactState,
)


@dataclass
class TrajectoryParams:
    dt: float = 0.02
    ramp_steps: int = 25
    hold_steps: int = 10
    unload_steps: int = 25


class ContactModel:
    """Hertzian contact: F = k * depth^1.5. Inverted to get the depth a
    force-controlled actuator would command to reach a target force. The
    sensor model uses depth_for_force() so this physics is actually exercised."""

    def __init__(self, hertz_k: float, indenter_radius_mm: float) -> None:
        self.k = float(hertz_k)
        self.radius = float(indenter_radius_mm)

    def depth_for_force(self, force_N: float) -> float:
        if force_N <= 0:
            return 0.0
        return float((force_N / self.k) ** (1.0 / 1.5))

    def force_for_depth(self, depth_mm: float) -> float:
        if depth_mm <= 0:
            return 0.0
        return float(self.k * depth_mm**1.5)


def force_trajectory(target_force_N: float, tp: TrajectoryParams):
    """Yield (t, applied_force, phase) over one press: ramp -> hold -> unload.

    A sub-threshold / zero target still yields a (flat) trajectory so the
    no-contact class is generated the same way as contact samples.
    """
    t = 0.0
    # Loading ramp (smooth, not linear, to look like a real actuator settle).
    for i in range(tp.ramp_steps):
        frac = (i + 1) / tp.ramp_steps
        f = target_force_N * (1 - np.cos(np.pi * frac)) / 2  # ease-in/out
        yield t, float(f), PHASE_RAMP
        t += tp.dt
    # Hold at target.
    for _ in range(tp.hold_steps):
        yield t, float(target_force_N), PHASE_HOLD
        t += tp.dt
    # Unloading ramp.
    for i in range(tp.unload_steps):
        frac = (i + 1) / tp.unload_steps
        f = target_force_N * (1 + np.cos(np.pi * frac)) / 2
        yield t, float(f), PHASE_UNLOAD
        t += tp.dt


def make_state(
    *,
    t: float,
    target_force_N: float,
    applied_force_N: float,
    contact_x: float,
    contact_y: float,
    phase: str,
    indenter_radius_mm: float,
    contact_threshold_N: float = 0.02,
) -> ContactState:
    return ContactState(
        t=t,
        target_force_N=target_force_N,
        applied_force_N=applied_force_N,
        contact_x=contact_x,
        contact_y=contact_y,
        in_contact=applied_force_N > contact_threshold_N,
        phase=phase,
        indenter_radius_mm=indenter_radius_mm,
    )
