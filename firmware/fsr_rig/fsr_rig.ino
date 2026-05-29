/*
 * fsr_rig.ino  --  real tactile rig firmware for the fibercal HAL.
 *
 * Implements the exact serial protocol documented in fibercal.hal.ArduinoRig:
 *
 *   host  -> board :  "F <target_force_N>\n"     command a target force
 *   board -> host  :  "<force_N> <ch0> <ch1> ...\n"   measured force + channels
 *
 * This lets a $10 Arduino + force-sensitive-resistor rig drop into the SAME
 * TactileRig interface as the simulator: the Python pipeline (dataset logging,
 * ML, characterization) is identical whether driven by SimRig or ArduinoRig.
 *
 * Wiring (each FSR is one "fiber channel" taxel):
 *   5V --- FSR --- A{i} --- 10k pulldown --- GND        (voltage divider)
 *   more FSR pressed -> lower resistance -> higher analog voltage.
 *
 * Map ADC -> pseudo-force with a one-point gain (replace with a real load-cell
 * calibration once an HX711 + load cell is added).
 */

const int N_CH = 4;                 // FSR channels on A0..A3
const int FSR_PINS[N_CH] = {A0, A1, A2, A3};
const float ADC_TO_FORCE = 10.0 / 1023.0;  // crude pseudo-force scale (N/count)

float targetForce = 0.0;            // last commanded target (logged, not actuated
                                    // here -- a stepper/servo loop would use it)

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N_CH; i++) pinMode(FSR_PINS[i], INPUT);
}

void loop() {
  // --- parse "F <target>\n" commands from the host ---
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.startsWith("F")) {
      targetForce = line.substring(1).toFloat();
    }
  }

  // --- read channels + estimate force, stream one line back ---
  int raw[N_CH];
  long sum = 0;
  for (int i = 0; i < N_CH; i++) {
    raw[i] = analogRead(FSR_PINS[i]);
    sum += raw[i];
  }
  float measuredForce = (sum / (float)N_CH) * ADC_TO_FORCE;  // proxy for load cell

  Serial.print(measuredForce, 4);
  for (int i = 0; i < N_CH; i++) {
    Serial.print(' ');
    Serial.print(raw[i] / 1023.0, 4);   // normalised channel reading
  }
  Serial.print('\n');

  delay(20);   // ~50 Hz, matches the sim trajectory dt = 0.02 s
}
