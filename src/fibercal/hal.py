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

    Protocol (newline-delimited ASCII, matches the sketch in firmware/):
        host -> board :  "F <target_force_N>\n"   command target force
        board -> host :  "<force_N> <ch0> <ch1> ...\n"   measured force + channels

    This stub documents the contract and connects if a board is present; it
    raises a clear error otherwise so the sim path stays the default.
    """

    def __init__(self, cfg: Config, port: str = "/dev/ttyACM0", baud: int = 115200):
        try:
            import serial  # pyserial, optional dependency
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "ArduinoRig needs pyserial: pip install pyserial"
            ) from e
        self._serial = serial
        self.port = port
        self.baud = baud
        self.cfg = cfg
        self.radius = float(cfg["indenter"]["radius_mm"])
        self._conn = None

    def reset(self) -> None:  # pragma: no cover - requires hardware
        if self._conn is None:
            self._conn = self._serial.Serial(self.port, self.baud, timeout=2)

    def press(self, x: float, y: float, target_force_N: float):  # pragma: no cover
        raise NotImplementedError(
            "Connect an Arduino+FSR rig and implement the serial read loop here. "
            "The contract is documented in this class' docstring and firmware/."
        )
