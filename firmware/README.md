# Real rig firmware (Arduino + FSR)

This is the **hardware path** of the same Hardware Abstraction Layer the
simulator uses. A real Arduino + force-sensitive-resistor (FSR) rig speaks the
exact serial protocol that `fibercal.hal.ArduinoRig` expects, so the entire
Python pipeline — dataset logging, inverse ML, characterization — runs
unchanged whether driven by `SimRig` or `ArduinoRig`. This is the proof that the
"simulation-first, hardware-ready" abstraction is real, not aspirational.

## Bill of materials (≈ $10)
- 1× Arduino Uno/Nano
- 1–4× FSR (e.g. Interlink 402), one per "fiber channel" taxel
- 1× 10 kΩ resistor per FSR (pulldown)
- (optional) HX711 + load cell for true ground-truth force

## Wiring (voltage divider, per FSR)
```
5V ──[ FSR ]──┬── A{i}        more pressure → lower FSR resistance
              │                → higher analog voltage at A{i}
            [10kΩ]
              │
             GND
```

## Protocol
```
host  -> board :  "F <target_force_N>\n"
board -> host  :  "<force_N> <ch0> <ch1> ...\n"   (newline-delimited ASCII, 115200 baud)
```

## Channel count: 4 (real) vs 64 (sim)
The sketch ships a **4-FSR proof of concept**; the simulator models a 64-channel
fiber array. This mismatch is intentional and harmless: the rig **reports** its
channel count, `ArduinoRig` learns it from the first reading, and `ml.py` /
`metrics.py` derive the channel columns from the data — so a 4-channel real
dataset and the 64-channel sim dataset both flow through the identical pipeline.
Scale up by adding FSRs (raise `N_CH` in the sketch) or wiring the real fiber
array's readout. FSRs do not sense position, so the commanded `(x, y)` from the
stage is logged as the ground-truth contact location.

## Bringing it online
1. Flash `fsr_rig/fsr_rig.ino`.
2. `ArduinoRig.press()` already implements the serial read loop (see `hal.py`).
3. Run with the real rig:  `python scripts/run_experiment.py --rig arduino --port /dev/ttyACM0`
   (no code change — the `--rig` flag swaps `SimRig` for `ArduinoRig` via the HAL).

The FSR's real hysteresis and drift then serve as a sanity check on the
simulator's forward-model assumptions (the sim deliberately includes
hysteresis/drift/saturation — compare the two).
