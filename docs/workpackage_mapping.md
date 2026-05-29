# Work-package mapping

How this project maps to the target ETH RSL project
*"Create a repeatable, precise, high-throughput robotic setup for characterizing a custom contact sensor."*

| Target work package / requirement | Where it lives here | Honest scope |
|---|---|---|
| **WP1 — Literature review of tactile-sensor calibration setups** | [`docs/lit_review.md`](lit_review.md) | Short, real references; the forward model and metrics follow them. |
| **WP2 — Characterization of resolution, sensitivity, range** | `metrics.py`, `viz.datasheet` → `figures/datasheet.png` | Sensitivity, resolution, range, hysteresis, repeatability, drift all computed and plotted. |
| **WP3 — Curation of a dataset of the sensor's response** | `sweeps.py` + `logger.py` → `data/dataset.parquet` | ~112k labelled rows, 4 sweep types, reproducible (seed + config hash). The dataset *is* a deliverable. |
| Robotic experimental setup / actuated test rig | `contact.py` (force-control state machine), `hal.py` (rig) | Virtual indenter under closed-loop force control. |
| Precise input↔output force/response pairs | dataset schema (`schema.py`) | Ground-truth `(force, x, y)` ↔ 64 channel readings per sample. |
| ML / AI for model training | `ml.py` (inverse model), `active_learning.py` | Detection + force + localization; baselines→MLP; uncertainty/coverage AL. |
| Automated calibration | `active_learning.py` | Core-set sample selection; sample-efficiency study. |
| Embedded programming / motor control (Arduino) | `firmware/` + `contact.py` force loop | The force-control loop is the logic that ports to an Arduino + stepper + load cell; FSR sketch provided. |
| Robotics & ROS | `rosmimic.py` | ROS2-style node/topic data flow; real `rclpy` port is a documented next step. |
| CAD / mechanical design | *partial gap* | Simulated geometry only. A real continuation would add a CAD test-rig model (linear stage + load cell + fixture). See roadmap. |

## How "simulation-based" is a strength (stated honestly)
The full data → model → characterization pipeline is built and tested in simulation **with reproducible, seeded data**. On day one with real hardware, only two blocks change — the sensor forward model (replaced by real readings) and the actuator driver — because everything downstream talks to the `TactileRig` HAL. The simulator also enables sweeping noise/design parameters that would be slow or impossible to vary on hardware. No hardware results are claimed.
