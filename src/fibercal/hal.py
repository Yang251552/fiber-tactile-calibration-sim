"""Hardware Abstraction Layer.

A TactileRig exposes the same interface whether driven by simulation or by a
real Arduino + force-sensitive-resistor rig. The whole data/ML/characterization
pipeline talks to this interface only, so swapping sim <-> real is a one-line
change. This is what makes the 'simulation-first, hardware-ready' story real
rather than aspirational.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import Config
from .contact import ContactModel, TrajectoryParams, force_trajectory, make_state
from .schema import ContactState
from .sensor_model import OpticalFiberSensorModel


class TactileRig(ABC):
    """Common interface for sim and real tactile calibration rigs."""

    n_channels: int
    channel_cols: list[str]

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def press(self, x: float, y: float, target_force_N: float):
        """Execute one full press at (x, y) to a target force.
        Yields (ContactState, channel_readings) over the press trajectory."""
        ...


class SimRig(TactileRig):
    """Fully simulated rig: analytic contact + optical-fiber forward model."""

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.contact = ContactModel(
            hertz_k=cfg["indenter"]["hertz_k"],
            indenter_radius_mm=cfg["indenter"]["radius_mm"],
        )
        self.sensor = OpticalFiberSensorModel(
            cfg["surface"], cfg["sensor"], self.rng,
            hertz_k=cfg["indenter"]["hertz_k"],
        )
        self.n_channels = self.sensor.n_channels
        self.channel_cols = self.sensor.channel_cols
        tj = cfg["trajectory"]
        self.tp = TrajectoryParams(
            dt=tj["dt"], ramp_steps=tj["ramp_steps"],
            hold_steps=tj["hold_steps"], unload_steps=tj["unload_steps"],
        )
        self.radius = float(cfg["indenter"]["radius_mm"])

    def reset(self) -> None:
        self.sensor.reset()

    def press(self, x: float, y: float, target_force_N: float):
        self.sensor.reset()  # fresh dynamic state -> independent press
        for t, f, phase in force_trajectory(target_force_N, self.tp):
            state = make_state(
                t=t, target_force_N=target_force_N, applied_force_N=f,
                contact_x=x, contact_y=y, phase=phase,
                indenter_radius_mm=self.radius,
            )
            channels = self.sensor.read(state)
            yield state, channels


class ArduinoRig(TactileRig):
    """Real rig over serial: Arduino + FSR(s) + (optional) load cell.

    Protocol (newline-delimited ASCII, matches firmware/fsr_rig/fsr_rig.ino):
        host -> board :  "F <target_force_N>\n"        command target force
        board -> host :  "<force_N> <ch0> <ch1> ...\n"  measured force + channels

    The number of channels is whatever the board reports (the firmware ships a
    4-FSR proof of concept; a real fiber sensor would report N). The rest of the
    pipeline derives channel columns from the data, so it is channel-count
    agnostic -- a 4-channel real dataset and the 64-channel sim dataset both flow
    through ml.py / metrics.py unchanged. FSRs do not sense position, so the
    commanded (x, y) is logged as ground-truth contact location (the operator /
    stage places the probe there).
    """

    def __init__(self, cfg: Config, port: str = "/dev/ttyACM0", baud: int = 115200,
                 n_steps: int = 20, settle_steps: int = 5):
        try:
            import serial  # pyserial, optional dependency
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("ArduinoRig needs pyserial: pip install pyserial") from e
        self._serial = serial
        self.port = port
        self.baud = baud
        self.cfg = cfg
        self.radius = float(cfg["indenter"]["radius_mm"])
        self.n_steps = n_steps
        self.settle_steps = settle_steps
        self._conn = None
        self.n_channels = None       # learned from the first reading
        self.channel_cols = None

    def reset(self) -> None:  # pragma: no cover - requires hardware
        if self._conn is None:
            self._conn = self._serial.Serial(self.port, self.baud, timeout=2)

    def _read_line(self):  # pragma: no cover - requires hardware
        from .schema import channel_columns
        raw = self._conn.readline().decode("ascii", "ignore").split()
        if len(raw) < 2:
            return None, None
        force = float(raw[0])
        channels = np.array([float(v) for v in raw[1:]])
        if self.n_channels is None:
            self.n_channels = len(channels)
            self.channel_cols = channel_columns(self.n_channels)
        return force, channels

    def press(self, x: float, y: float, target_force_N: float):  # pragma: no cover
        from .contact import make_state
        from .schema import PHASE_HOLD, PHASE_RAMP
        self.reset()
        self._conn.write(f"F {target_force_N}\n".encode("ascii"))
        for i in range(self.n_steps):
            force, channels = self._read_line()
            if force is None:
                continue
            phase = PHASE_HOLD if i >= self.settle_steps else PHASE_RAMP
            state = make_state(
                t=i * 0.02, target_force_N=target_force_N, applied_force_N=force,
                contact_x=x, contact_y=y, phase=phase,
                indenter_radius_mm=self.radius,
            )
            yield state, channels
