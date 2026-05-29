"""Optical-fiber tactile sensor forward model.

Maps ground-truth contact (location + force, with loading history) to a vector
of N noisy fiber-channel readings. This is the fictional sensor's "physics" and
the scientific core of the project. The ML side never sees these parameters --
it must learn the inverse from data alone.

Modelled effects (all toggle-able via config):
  * spatial response : Gaussian load-spreading kernel over an 8x8 taxel grid,
                       width set by the Hertzian contact patch (force/radius)
  * nonlinearity     : tanh saturation (optical attenuation saturates)
  * crosstalk        : near-identity mixing matrix between channels
  * response delay   : first-order lag (sensor cannot track force instantly)
  * hysteresis       : extra gain on unloading -> opens a loop vs loading
  * drift            : slow per-channel random walk

Note on drift: drift is reset per press (see SimRig.press), so the *dataset*
exhibits only within-press drift; the intrinsic slow drift is characterized
separately by metrics.drift_probe, which drives this model without reset.
"""
from __future__ import annotations

import numpy as np

from .schema import PHASE_UNLOAD, ContactState, channel_columns


class OpticalFiberSensorModel:
    def __init__(self, surface_cfg: dict, sensor_cfg: dict, rng: np.random.Generator,
                 hertz_k: float = 2.0):
        self.rng = rng
        self.hertz_k = float(hertz_k)  # couples Hertzian indentation -> spread
        size = float(surface_cfg["size_mm"])
        n = int(surface_cfg["taxels_per_side"])
        margin = float(surface_cfg["margin_mm"])
        self.n_side = n
        self.n_channels = n * n
        self.channel_cols = channel_columns(self.n_channels)

        # Taxel (fiber pickup) positions on the surface.
        coords = np.linspace(margin, size - margin, n)
        gx, gy = np.meshgrid(coords, coords)
        self.taxel_xy = np.column_stack([gx.ravel(), gy.ravel()])  # (C, 2)

        # Forward-model parameters.
        self.sigma = float(sensor_cfg["spatial_sigma_mm"])
        self.gain = float(sensor_cfg["gain"])
        self.saturation = float(sensor_cfg["saturation"])
        self.noise_std = float(sensor_cfg["noise_std"])
        self.drift_rate = float(sensor_cfg["drift_rate"])
        self.hysteresis = float(sensor_cfg["hysteresis"])
        self.lag_alpha = float(sensor_cfg["lag_alpha"])
        self.adc_bits = int(sensor_cfg["adc_bits"])
        self.baseline = float(sensor_cfg["baseline"])

        # Fixed crosstalk matrix M = I + eps * R (small random off-diagonal).
        eps = float(sensor_cfg["crosstalk"])
        R = rng.normal(0.0, 1.0, size=(self.n_channels, self.n_channels))
        np.fill_diagonal(R, 0.0)
        self.M = np.eye(self.n_channels) + eps * R

        # Per-channel static gain/offset imperfections (manufacturing spread).
        self.chan_gain = rng.normal(1.0, 0.05, size=self.n_channels)
        self.chan_offset = rng.normal(0.0, 0.005, size=self.n_channels)

        self.reset()

    def reset(self) -> None:
        """Reset dynamic state (lag filter + drift). Call between presses for
        independent samples; keep across a press to expose delay/hysteresis."""
        self._filtered = np.full(self.n_channels, self.baseline)
        self._drift = np.zeros(self.n_channels)

    def _effective_sigma(self, force: float, radius_mm: float) -> float:
        """Load-spread width grows with the Hertzian contact patch.

        Indentation depth d = (F/k)^(2/3); Hertzian contact radius a = sqrt(R*d).
        The silicone spreads load over the base sigma convolved with the contact
        patch: sigma_eff = sqrt(sigma^2 + a^2). This makes both the Hertzian
        model and the indenter radius physically active, not decorative."""
        if force <= 0:
            return self.sigma
        depth = (force / self.hertz_k) ** (2.0 / 3.0)
        a = np.sqrt(max(radius_mm, 1e-6) * depth)  # Hertzian contact radius (mm)
        return float(np.sqrt(self.sigma**2 + a**2))

    def _spatial_activation(self, x: float, y: float, force: float,
                            radius_mm: float) -> np.ndarray:
        """Gaussian-weighted load spreading -> per-channel raw activation.
        Spread width is set by the Hertzian contact patch (force/radius dep.)."""
        sigma_eff = self._effective_sigma(force, radius_mm)
        d2 = np.sum((self.taxel_xy - np.array([x, y])) ** 2, axis=1)
        w = np.exp(-d2 / (2.0 * sigma_eff**2))
        return w * force

    def read(self, state: ContactState) -> np.ndarray:
        """One sensor reading for the given ground-truth contact state."""
        a = self._spatial_activation(state.contact_x, state.contact_y,
                                     state.applied_force_N,
                                     state.indenter_radius_mm)

        # Hysteresis: unloading reads higher than loading at equal force.
        g = self.gain * (1.0 + self.hysteresis) if state.phase == PHASE_UNLOAD else self.gain

        # Nonlinear saturation.
        r = g * np.tanh(a / self.saturation)

        # Crosstalk mixing.
        s = self.M @ r

        # Per-channel imperfections + baseline.
        s = self.chan_gain * s + self.chan_offset + self.baseline

        # First-order response delay (lag filter, stateful across the press).
        self._filtered = self.lag_alpha * self._filtered + (1 - self.lag_alpha) * s

        # Slow drift random walk.
        self._drift += self.rng.normal(0.0, self.drift_rate, size=self.n_channels)
        out = self._filtered + self._drift

        # Read noise.
        out = out + self.rng.normal(0.0, self.noise_std, size=self.n_channels)

        # 12-bit ADC quantisation over a fixed [0, 2] range.
        levels = 2**self.adc_bits
        out = np.clip(out, 0.0, 2.0)
        out = np.round(out / 2.0 * (levels - 1)) / (levels - 1) * 2.0
        return out
