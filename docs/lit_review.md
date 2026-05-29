# Literature review (brief) — tactile-sensor calibration setups

A short survey of how custom contact/tactile sensors are characterized and
calibrated, and how this project's design follows that practice.

## Calibration rigs: controlled force → recorded response
The standard approach applies *known, controlled* contacts and records sensor
output to build input↔output pairs. Rigs use a motorized linear stage or robot
arm with a load cell for ground-truth force and an indenter of known geometry,
sweeping force levels and contact locations. This project mirrors that with a
force-controlled virtual indenter (`contact.py`) and grid/random/ramp/repeat
sweeps (`sweeps.py`).

## Learning the inverse (readings → force, location)
For soft and optical tactile sensors the readings-to-contact map is nonlinear
and coupled, so it is commonly learned from data:
- **Optical-fiber / FBG sensors** + ML recover force magnitude and contact
  location with high fidelity (reported force R² ≈ 0.97, localization error of a
  few mm). Our simulated MLP reaches force R² ≈ 0.99 and ~0.8 mm median
  localization — better than hardware because the sim is cleaner; the *shape* of
  the result matches the literature.
- Linear/ridge baselines are standard sanity checks; a learned model must beat
  them (it does here).

## Characterization metrics
Sensor datasheets report **sensitivity** (response per unit force),
**resolution** (smallest distinguishable force step, often a k·σ rule from
repeat measurements), **dynamic range** (detection floor to saturation),
**hysteresis** (loading vs unloading gap, % of full scale), **repeatability**
and **drift**. `metrics.py` computes all of these and `viz.datasheet` renders a
one-page datasheet.

## Sim-to-real and few-sample calibration
Reducing the number of physical calibration probes (active learning, domain
adaptation, self-calibration) is an active theme. This motivates the core-set
active-calibration study (`active_learning.py`): on a realistically biased probe
pool, coverage-based selection reaches target accuracy with far fewer samples.

## Forward-model design choices grounded in the above
The synthetic optical-fiber model includes the effects real such sensors show:
load spreading through the compliant medium (Gaussian kernel, width set by the
Hertzian contact patch), nonlinear saturation of the optical signal, inter-
channel crosstalk, response lag, slow drift, and quantisation noise — so the
learned inverse and the characterization numbers are non-trivial.

*References are intentionally summarized rather than cited verbatim; this file is
a design rationale, not a formal survey.*
