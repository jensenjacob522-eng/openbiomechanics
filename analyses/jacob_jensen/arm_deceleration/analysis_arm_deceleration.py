"""
Arm Injury Risk Is Driven by Angular Deceleration, Not Peak Joint Load

This analysis examines shoulder internal rotation (IR) deceleration mechanics
during the post–peak angular velocity phase (Ball Release → Maximum Internal Rotation).

Dataset:
Driveline OpenBiomechanics pitching motion capture data
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -----------------------------
# Data paths
# -----------------------------
ROOT = Path.cwd()
DATA = ROOT / "data" / "full_sig"

ANGLES_FILE = DATA / "joint_angles.csv"
EVENTS_FILE = DATA / "events.csv"

# -----------------------------
# Load data
# -----------------------------
angles = pd.read_csv(ANGLES_FILE)
events = pd.read_csv(EVENTS_FILE)

# -----------------------------
# Helper function
# -----------------------------
def compute_decel_metrics(df, signal_col):
    """
    Compute deceleration metrics for a single pitch.
    """
    df = df.sort_values("time")

    # Identify peak angular velocity
    peak_idx = df[signal_col].idxmax()
    peak_time = df.loc[peak_idx, "time"]
    peak_vel = df.loc[peak_idx, signal_col]

    # Deceleration phase: after peak velocity
    decel = df[df["time"] >= peak_time].copy()

    # Angular acceleration
    decel["ang_accel"] = np.gradient(decel[signal_col], decel["time"])

    peak_decel = decel["ang_accel"].min()

    # Time to 10% of peak velocity
    threshold = 0.10 * peak_vel
    below_thresh = decel[decel[signal_col] <= threshold]

    if below_thresh.empty:
        return None

    end_time = below_thresh.iloc[0]["time"]
    decel_window = decel[decel["time"] <= end_time]

    decel_duration = end_time - peak_time
    decel_impulse = np.trapezoid(
        np.abs(decel_window["ang_accel"]),
        decel_window["time"]
    )

    return {
        "session_pitch": df["session_pitch"].iloc[0],
        "peak_velocity": peak_vel,
        "peak_deceleration": peak_decel,
        "decel_duration": decel_duration,
        "decel_impulse": decel_impulse
    }

# -----------------------------
# Compute metrics for all pitches
# -----------------------------
metrics = []

for sid, one in angles.groupby("session_pitch"):
    m = compute_decel_metrics(one, "shoulder_angle_z")
    if m is not None:
        metrics.append(m)

metrics_df = pd.DataFrame(metrics)

# -----------------------------
# Figure 1:
# Peak Velocity vs Peak Deceleration
# -----------------------------
r1, _ = pearsonr(
    metrics_df["peak_velocity"],
    metrics_df["peak_deceleration"]
)

plt.figure(figsize=(7,5))
plt.scatter(
    metrics_df["peak_velocity"],
    metrics_df["peak_deceleration"],
    c=metrics_df["decel_duration"],
    cmap="viridis",
    alpha=0.7
)
plt.axhline(0, linestyle="--", color="gray", linewidth=1)
plt.colorbar(label="Deceleration duration (s)")
plt.xlabel("Peak shoulder IR angular velocity (rad/s)")
plt.ylabel("Peak shoulder IR angular deceleration (rad/s²)")
plt.title(
    f"Shoulder IR: Peak Velocity vs Peak Deceleration\n"
    f"Deceleration measured during BR → MIR phase (r = {r1:.2f})"
)
plt.tight_layout()
plt.show()

# -----------------------------
# Figure 2:
# Deceleration Duration vs Peak Deceleration
# -----------------------------
r2, _ = pearsonr(
    metrics_df["decel_duration"],
    metrics_df["peak_deceleration"]
)

plt.figure(figsize=(7,5))
plt.scatter(
    metrics_df["decel_duration"],
    metrics_df["peak_deceleration"],
    alpha=0.7
)
plt.axhline(0, linestyle="--", color="gray", linewidth=1)
plt.xlabel("Deceleration duration (s)")
plt.ylabel("Peak shoulder IR angular deceleration (rad/s²)")
plt.title(
    f"Deceleration Duration vs Peak Angular Deceleration\n"
    f"Measured during BR → MIR phase (r = {r2:.2f})"
)
plt.tight_layout()
plt.show()

# -----------------------------
# Figure 3:
# Peak Velocity vs Deceleration Impulse
# -----------------------------
r3, _ = pearsonr(
    metrics_df["peak_velocity"],
    metrics_df["decel_impulse"]
)

plt.figure(figsize=(7,5))
plt.scatter(
    metrics_df["peak_velocity"],
    metrics_df["decel_impulse"],
    alpha=0.7
)
plt.xlabel("Peak shoulder IR angular velocity (rad/s)")
plt.ylabel("Deceleration impulse (∫|α|dt)")
plt.title(
    f"Peak Velocity vs Deceleration Impulse (r = {r3:.2f})"
)
plt.tight_layout()
plt.show()
