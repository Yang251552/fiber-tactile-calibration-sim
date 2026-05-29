# Roadmap — from simulation to hardware

This project is the simulation-first foundation. The interfaces were designed so
the transition to real hardware touches as little code as possible.

## Done (this repo)
- ROS-style node/topic architecture, config-hash reproducibility, schema-as-contract.
- Hertzian force-control state machine + optical-fiber forward sensor model.
- 4-sweep dataset generation; leakage-safe (press-grouped) train/test splits.
- Inverse models (linear baseline → MLP) for detection / force / localization.
- Sensor characterization datasheet (sensitivity / resolution / range / hysteresis / repeatability / drift).
- Core-set active-calibration sample-efficiency study.
- Interactive dashboard; Arduino + FSR firmware implementing the HAL protocol.

## Next: hardware bring-up (smallest change first)
1. **Finish `ArduinoRig.press()`** — read the serial loop from the FSR sketch.
   Everything downstream (logging, ML, characterization) already works via the HAL.
2. **Single-channel real validation** — one FSR; collect a small real dataset;
   compare its measured hysteresis/drift against the simulator's modelled values.
3. **Ground-truth force** — add an HX711 + load cell for true applied force,
   replacing the ADC pseudo-force proxy.

## Next: modelling / ML
- Real `rclpy` node port (publish `/sensor_readings`, view in `ros2 topic echo` / `rqt`).
- Uncertainty-aware inverse model (GP or MLP ensemble) reporting per-prediction confidence.
- Active calibration *in the loop* on hardware: the rig chooses the next probe online.

## Next: mechanical (the current gap)
- CAD model of a real test rig (motorized linear stage + load cell + sensor fixture),
  exported renders into the README — closes the CAD/mechanical-design requirement.
