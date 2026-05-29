"""Interactive demo dashboard (for the README GIF).

    streamlit run app/dashboard.py

Drag the contact location / force and watch the simulated 8x8 optical-fiber
response live; if a dataset exists, a small trained model shows predicted vs
true force.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fibercal.config import load_config
from fibercal.contact import make_state
from fibercal.hal import SimRig
from fibercal.logger import channel_cols, load_dataset
from fibercal.schema import PHASE_HOLD

st.set_page_config(page_title="Fiber Tactile Calibration Sim", layout="wide")


@st.cache_resource
def get_rig():
    cfg = load_config()
    return cfg, SimRig(cfg)


@st.cache_resource
def get_force_model():
    """Quick force regressor trained on the dataset, if present."""
    data_path = ROOT / "data" / "dataset.parquet"
    if not data_path.exists():
        return None
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = load_dataset(data_path)
    df = df[df["in_contact"]]
    if len(df) > 4000:
        df = df.sample(4000, random_state=0)
    cols = channel_cols(df)
    model = make_pipeline(StandardScaler(),
                          MLPRegressor((128, 64), max_iter=300, random_state=0))
    model.fit(df[cols].values, df["applied_force_N"].values)
    return model, cols


cfg, rig = get_rig()
size = float(cfg["surface"]["size_mm"])

st.title("🔬 Simulated Optical-Fiber Tactile Sensor — Calibration Rig")
st.caption("A virtual robotic calibration setup: an indenter presses a sensorized "
           "patch; the 8×8 fiber array responds via a forward model "
           "(load-spreading · saturation · crosstalk · noise · hysteresis · lag). "
           "An inverse model recovers force & location.")

with st.sidebar:
    st.header("Contact probe")
    x = st.slider("x (mm)", 0.0, size, size / 2, 0.5)
    y = st.slider("y (mm)", 0.0, size, size / 2, 0.5)
    force = st.slider("applied force (N)", 0.0, 10.0, 5.0, 0.1)

# Steady-state reading at the chosen contact.
rig.sensor.reset()
state = make_state(t=0, target_force_N=force, applied_force_N=force,
                   contact_x=x, contact_y=y, phase=PHASE_HOLD,
                   indenter_radius_mm=rig.radius)
ch = rig.sensor.read(state)
n = rig.sensor.n_side

col1, col2 = st.columns(2)

with col1:
    st.subheader("8×8 fiber response")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(ch.reshape(n, n), origin="lower", cmap="inferno",
                   extent=[0, size, 0, size], vmin=0)
    ax.plot(x, y, "c+", markersize=16, markeredgewidth=2.5)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, label="reading")
    st.pyplot(fig)

with col2:
    st.subheader("Per-channel readings")
    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    ax2.bar(range(len(ch)), ch)
    ax2.set_xlabel("channel"); ax2.set_ylabel("reading")
    st.pyplot(fig2)

st.subheader("Ground truth vs inverse model")
mcols = st.columns(4)
mcols[0].metric("True force", f"{force:.2f} N")
mcols[1].metric("Contact", f"({x:.0f}, {y:.0f}) mm")
mcols[2].metric("In contact", "yes" if state.in_contact else "no")

fm = get_force_model()
if fm is not None:
    model, cols = fm
    pred = float(model.predict(ch.reshape(1, -1))[0])
    mcols[3].metric("Predicted force", f"{pred:.2f} N", f"{pred - force:+.2f} N")
else:
    mcols[3].info("Run scripts/run_experiment.py to enable the trained model.")
