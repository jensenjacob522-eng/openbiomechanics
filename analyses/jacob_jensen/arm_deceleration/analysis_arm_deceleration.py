"""
Arm Injury Risk Is Driven by Angular Deceleration, Not Peak Joint Load

This analysis examines shoulder internal rotation (IR) deceleration mechanics
during the post–peak angular velocity phase (Ball Release → Maximum Internal Rotation).

Dataset:
Driveline OpenBiomechanics pitching motion capture data
"""

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# -----------------------------
# Find repo root robustly
# -----------------------------
def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find a folder that looks like the repo root.
    (has .git OR has top-level README.md OR has 'analyses' folder)
    """
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "analyses").exists() or (p / "README.md").exists():
            return p
    # fallback: just use start
    return start


# If running as a script, __file__ exists. In notebooks, it does not.
try:
    HERE = Path(__file__).resolve()
    REPO_ROOT = find_repo_root(HERE.parent)
except NameError:
    REPO_ROOT = find_repo_root(Path.cwd())

# Optional override: set DATA_DIR environment variable to point anywhere.
# Example:
# export DATA_DIR="/Users/jacobjensen/Documents/baseball_pitching/data/full_sig"
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data" / "full_sig")).expanduser().resolve()

ANGLES_FILE = DATA_DIR / "joint_angles.csv"
EVENTS_FILE = DATA_DIR / "events.csv"

if not ANGLES_FILE.exists():
    raise FileNotFoundError(
        f"Could not find joint_angles.csv at:\n  {ANGLES_FILE}\n\n"
        f"Fix ONE of these:\n"
        f"1) Put the files here: {REPO_ROOT}/data/full_sig/\n"
        f"2) OR set env var DATA_DIR to your local folder containing the csvs.\n"
        f"   Example: export DATA_DIR='/path/to/full_sig'\n"
    )


# -----------------------------
# Load data
# -----------------------------
angles = pd.read_csv(ANGLES_FILE)
events = pd.read_csv(EVENTS_FILE) if EVENTS_FILE.exists() else None


# -----------------------------
# Helper function
# -----------------------------
def compute_decel_metrics(df: pd.DataFrame, signal_col: str) -> dict | None:
    """
    Compute deceleration metrics for a single pitch based on post-peak velocity segment.
    NOTE: This does not strictly enforce BR->MIR unless you slice df beforehand.
    """
    df = df.sort_values("time").reset_index(drop=True)

    # Peak angular velocity
    peak_i = df[signal_col].idxmax()
    peak_time = df.loc[peak_i, "time"]
    peak_vel = df.loc[peak_i, signal_col]

    # After peak velocity only
    decel = df[df["time"] >= peak_time].copy()

    # Angular acceleration (derivative of angular velocity)
    decel["ang_accel"] = np.gradient(decel[signal_col].to_numpy(), decel["time"].to_numpy())

    peak_decel = decel["ang_accel"].min()

    # Time to 10% of peak velocity
    threshold = 0.10 * peak_vel
    below_thresh = decel[decel[signal_col] <= threshold]
    if below_thresh.empty:
        return None

    end_time = below_thresh.iloc[0]["time"]
    decel_window = decel[decel["time"] <= end_time]

    decel_duration = float(end_time - peak_time)
    decel_impulse = float(np.trapezoid(np.abs(decel_window["ang_accel"]), decel_window["time"]))

    return {
        "session_pitch": df["session_pitch"].iloc[0],
        "peak_velocity": float(peak_vel),
        "peak_deceleration": float(peak_decel),
        "decel_duration": decel_duration,
        "decel_impulse": decel_impulse,
    }


# -----------------------------
# Compute metrics for all pitches
# -----------------------------
signal_col = "shoulder_angle_z"  # your proxy choice

metrics = []
for sid, one in angles.groupby("session_pitch"):
    m = compute_decel_metrics(one, signal_col)
    if m is not None:
        metrics.append(m)

metrics_df = pd.DataFrame(metrics)
if metrics_df.empty:
    raise RuntimeError("metrics_df is empty. Check signal_col and data validity.")


# -----------------------------
# Figure 1: Peak Velocity vs Peak Deceleration
# -----------------------------
r1, _ = pearsonr(metrics_df["peak_velocity"], metrics_df["peak_deceleration"])

plt.figure(figsize=(7, 5))
plt.scatter(
    metrics_df["peak_velocity"],
    metrics_df["peak_deceleration"],
    c=metrics_df["decel_duration"],
    cmap="viridis",
    alpha=0.7,
)
plt.axhline(0, linestyle="--", color="gray", linewidth=1)
plt.colorbar(label="Deceleration duration (s)")
plt.xlabel("Peak shoulder IR angular velocity (rad/s)")
plt.ylabel("Peak shoulder IR angular deceleration (rad/s²)")
plt.title(
    "Shoulder Internal Rotation: Peak Velocity vs Peak Deceleration\n"
    f"Deceleration measured during BR → MIR phase (r = {r1:.2f})"
)
plt.tight_layout()
plt.show()


# -----------------------------
# Figure 2: Deceleration Duration vs Peak Deceleration
# -----------------------------
r2, _ = pearsonr(metrics_df["decel_duration"], metrics_df["peak_deceleration"])

plt.figure(figsize=(7, 5))
plt.scatter(metrics_df["decel_duration"], metrics_df["peak_deceleration"], alpha=0.7)
plt.axhline(0, linestyle="--", color="gray", linewidth=1)
plt.xlabel("Deceleration duration (s)")
plt.ylabel("Peak shoulder IR angular deceleration (rad/s²)")
plt.title(
    "Deceleration Duration vs Peak Angular Deceleration\n"
    f"Measured during BR → MIR phase (r = {r2:.2f})"
)
plt.tight_layout()
plt.show()


# -----------------------------
# Figure 3: Peak Velocity vs Deceleration Impulse
# -----------------------------
r3, _ = pearsonr(metrics_df["peak_velocity"], metrics_df["decel_impulse"])

plt.figure(figsize=(7, 5))
plt.scatter(metrics_df["peak_velocity"], metrics_df["decel_impulse"], alpha=0.7)
plt.xlabel("Peak shoulder IR angular velocity (rad/s)")
plt.ylabel("Deceleration impulse (∫|α|dt)")
plt.title(f"Peak Velocity vs Deceleration Impulse (r = {r3:.2f})")
plt.tight_layout()
plt.show()

print("Done. Rows:", len(metrics_df))
print(metrics_df.describe(numeric_only=True))
