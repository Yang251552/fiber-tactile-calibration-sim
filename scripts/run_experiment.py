#!/usr/bin/env python3
"""Drive the (sim) calibration rig and write a dataset.

    python scripts/run_experiment.py [--config config/experiment.yaml]
                                     [--out data/dataset.parquet]

Wires the rig through the ROS-style node graph and streams rows to the Logger.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fibercal.config import load_config
from fibercal.hal import SimRig
from fibercal.logger import write_dataset
from fibercal.rosmimic import Bus, topic_graph
from fibercal.sweeps import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/dataset.parquet")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    print(f"[config] {cfg.path.name}  sha={cfg.sha}  seed={cfg.seed}")

    rig = SimRig(cfg)
    print(f"[rig] SimRig  channels={rig.n_channels}")

    # ROS-style data flow (declared for the topic graph / architecture diagram).
    bus = Bus()
    for t in ("/cmd_force", "/contact_state", "/sensor_readings", "/dataset"):
        bus.subscribe(t, lambda m: None)

    t0 = time.time()
    rows = generate_dataset(rig, cfg)
    out = root / args.out
    df = write_dataset(rows, out)
    dt = time.time() - t0

    print(f"[topics] {topic_graph(bus)}")
    print(f"[dataset] {len(df):,} rows x {df.shape[1]} cols  ->  {out}")
    print(f"[dataset] presses: {df['press_id'].nunique():,}")
    print(f"[dataset] sweeps: " +
          ", ".join(f"{k}={v}" for k, v in df['sweep_type'].value_counts().items()))
    print(f"[done] {dt:.1f}s  ({len(df)/dt:,.0f} samples/s)")


if __name__ == "__main__":
    main()
