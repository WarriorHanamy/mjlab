"""Post-episode matplotlib plotter for trajectory tracking results.

Records actual and reference position/velocity during play via step callback
and generates a 3-panel matplotlib figure after *record_duration* s of data
(optionally skipping an initial *record_offset* s).

Usage::

    plotter = TrackPlotter(env.unwrapped)
    viewer.add_step_callback(plotter._on_step)
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Pick a backend that actually works in this environment.
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
  matplotlib.use("TkAgg")
else:
  matplotlib.use("Agg")

_POS_ERR_COLOR = "#1f77b4"
_VEL_ERR_COLOR = "#d62728"


class TrackPlotter:
  """Record tracking data and generate a 3-panel matplotlib figure.

  Panels:
    1. 2D XY trajectory — reference vs actual
    2. Position norm error over time (with mean / max annotation)
    3. Velocity norm error over time
  """

  def __init__(
    self,
    env,
    record_offset: float = 5.0,
    record_duration: float = 20.0,
  ) -> None:
    self._env = env
    self._record_offset = record_offset
    self._record_duration = record_duration
    self._done = False
    self._elapsed: float = 0.0

    # Accumulators (relative to offset start).
    self._t: list[float] = []
    self._pos_act_x: list[float] = []
    self._pos_act_y: list[float] = []
    self._pos_ref_x: list[float] = []
    self._pos_ref_y: list[float] = []
    self._pos_err: list[float] = []
    self._vel_err: list[float] = []

    self._fig: plt.Figure | None = None

  # -- public API --------------------------------------------------------

  def _on_step(self) -> None:
    """Step callback — skip *record_offset*, then record *record_duration*."""
    if self._done:
      return

    self._elapsed += self._env.step_dt

    if self._elapsed < self._record_offset:
      return

    try:
      traj = self._env.command_manager._terms["traj_command"]
      asset = self._env.scene["robot"]
    except (KeyError, AttributeError):
      return

    window_t = self._elapsed - self._record_offset

    pos_act = asset.data.root_link_pos_w[0, :3]
    vel_act = asset.data.root_link_lin_vel_w[0, :3]
    pos_ref = traj.command[0, :3]
    vel_ref = traj.ref_vel_w[0, :3]

    self._t.append(window_t)
    self._pos_act_x.append(float(pos_act[0]))
    self._pos_act_y.append(float(pos_act[1]))
    self._pos_ref_x.append(float(pos_ref[0]))
    self._pos_ref_y.append(float(pos_ref[1]))
    self._pos_err.append(float(torch.norm(pos_act - pos_ref)))
    self._vel_err.append(float(torch.norm(vel_act - vel_ref)))

    if window_t >= self._record_duration:
      self._done = True
      self._generate_plot()

  def close(self) -> None:
    """Close the matplotlib figure and release resources."""
    if self._fig is not None:
      plt.close(self._fig)
      self._fig = None

  # -- plot generation --------------------------------------------------

  def _generate_plot(self) -> None:
    """Build, save, and optionally show the 3-panel figure."""
    if self._fig is not None:
      return

    pos_err_arr = np.array(self._pos_err)
    vel_err_arr = np.array(self._vel_err)

    mean_pos = float(np.mean(pos_err_arr))
    max_pos = float(np.max(pos_err_arr))
    mean_vel = float(np.mean(vel_err_arr))
    max_vel = float(np.max(vel_err_arr))

    # Print stats to console.
    print()
    print(
      f"[TrackPlotter] === Tracking Results ({self._record_duration:.0f} s, "
      f"offset {self._record_offset:.0f} s) ==="
    )
    print(f"  Position error — Mean: {mean_pos:.4f} m, Max: {max_pos:.4f} m")
    print(f"  Velocity error — Mean: {mean_vel:.4f} m/s, Max: {max_vel:.4f} m/s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), num="Trajectory Tracking Results")
    self._fig = fig

    # --- Panel 1: 2D XY Trajectory ---
    ax0 = axes[0]
    ax0.plot(
      self._pos_ref_x,
      self._pos_ref_y,
      "--",
      color=_POS_ERR_COLOR,
      lw=1.5,
      alpha=0.7,
      label="Reference",
    )
    ax0.plot(
      self._pos_act_x,
      self._pos_act_y,
      "-",
      color=_VEL_ERR_COLOR,
      lw=1.5,
      alpha=0.9,
      label="Actual",
    )
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")
    ax0.set_title("2D XY Trajectory")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_aspect("equal", adjustable="datalim")

    # --- Panel 2: Position Norm Error ---
    ax1 = axes[1]
    ax1.plot(self._t, self._pos_err, "-", color=_POS_ERR_COLOR, lw=1.0)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position Error [m]")
    ax1.set_title("Position Norm Error")
    ax1.grid(True, alpha=0.3)
    ax1.text(
      0.95,
      0.95,
      f"Mean: {mean_pos:.3f} m\nMax: {max_pos:.3f} m",
      transform=ax1.transAxes,
      ha="right",
      va="top",
      fontsize=8,
      bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    # --- Panel 3: Velocity Norm Error ---
    ax2 = axes[2]
    ax2.plot(self._t, self._vel_err, "-", color=_VEL_ERR_COLOR, lw=1.0)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Velocity Error [m/s]")
    ax2.set_title("Velocity Norm Error")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.0)

    # Save to file (always).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"track_plot_{timestamp}.png"
    fig.savefig(save_name, dpi=150)
    print(f"[TrackPlotter] Saved: {os.path.abspath(save_name)}")

    # Non-blocking show (if a GUI backend is active).
    if matplotlib.get_backend() != "Agg":
      fig.show()
      fig.canvas.draw_idle()
      fig.canvas.flush_events()
