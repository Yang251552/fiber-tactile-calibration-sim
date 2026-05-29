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

## Bringing it online
1. Flash `fsr_rig/fsr_rig.ino`.
2. In `fibercal/hal.py`, finish `ArduinoRig.press()` (read the serial line loop)
   — the contract is fully documented in that class.
3. Swap `SimRig(cfg)` → `ArduinoRig(cfg, port=...)` in `scripts/run_experiment.py`.

The FSR's real hysteresis and drift then serve as a sanity check on the
simulator's forward-model assumptions (the sim deliberately includes
hysteresis/drift/saturation — compare the two).
