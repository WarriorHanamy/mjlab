"""Real-time matplotlib plotter for VTOL action-response dynamics.

Plots commanded vs actual (response) for thrust and angular velocity
components, reading directly from the environment's action term.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
  from matplotlib.axes import Axes
  from matplotlib.figure import Figure
  from matplotlib.lines import Line2D

try:
  matplotlib.use("TkAgg")  # Non-blocking window compatible with both viewers
except ImportError:
  matplotlib.use("Agg")  # Fallback for headless environments

_CMD_COLOR = "#1f77b4"  # blue
_ACT_COLOR = "#d62728"  # red
_HISTORY = 500
_UPDATE_INTERVAL = 3  # redraw every N calls


class ActionResponsePlotter:
  """Plots commanded vs actual thrust and angular velocity (body frame).

  Creates a separate non-blocking matplotlib figure with 4 subplots:
    1. Thrust [N]       — cmd vs act
    2. Ang Vel X [rad/s] — cmd vs act
    3. Ang Vel Y [rad/s] — cmd vs act
    4. Ang Vel Z [rad/s] — cmd vs act

  Usage:
      plotter = ActionResponsePlotter(env)
      viewer.add_step_callback(plotter._on_step)
      # Toggle with plotter.toggle() or bind to a key/button.
  """

  def __init__(self, env) -> None:
    self._env = env
    self._visible = False
    self._call_count = 0

    # Ring buffers — one per subplot, separate cmd/act
    self._t: collections.deque[float] = collections.deque(maxlen=_HISTORY)
    self._thr_cmd: collections.deque[float] = collections.deque(maxlen=_HISTORY)
    self._thr_act: collections.deque[float] = collections.deque(maxlen=_HISTORY)

    self._av_cmd: list[collections.deque[float]] = [
      collections.deque(maxlen=_HISTORY) for _ in range(3)
    ]
    self._av_act: list[collections.deque[float]] = [
      collections.deque(maxlen=_HISTORY) for _ in range(3)
    ]

    self._fig: Figure | None = None
    self._axes: list[Axes] = []
    self._ln_cmd: list[Line2D] = []
    self._ln_act: list[Line2D] = []

  # -- public API --------------------------------------------------------

  def toggle(self) -> None:
    if self._visible:
      self._hide()
    else:
      self._show()

  def _on_step(self) -> None:
    """Callback invoked after each physics step (registered on viewer)."""
    if not self._visible:
      return

    if self._fig is None:
      self._create_figure()
    if self._fig is None:
      return

    self._call_count += 1
    self._push_data()

    if self._call_count % _UPDATE_INTERVAL != 0:
      return
    self._redraw()

  def close(self) -> None:
    if self._fig is not None:
      plt.close(self._fig)
      self._fig = None

  @property
  def visible(self) -> bool:
    return self._visible

  # -- internals ---------------------------------------------------------

  def _resolve_term(self) -> Any:
    try:
      term = self._env.action_manager.get_term("control_action")
    except KeyError:
      return None
    # Unwrap PathTrackingAction wrapper (track task)
    if hasattr(term, "_control"):
      term = term._control
    if not (hasattr(term, "_thrust_dynamics") and hasattr(term, "_ang_vel_dynamics")):
      return None
    return term

  def _push_data(self) -> None:
    term = self._resolve_term()
    if term is None:
      return

    env_idx = 0
    pa = term.processed_actions  # type: ignore[union-attr]
    td = term._thrust_dynamics  # type: ignore[union-attr]
    avd = term._ang_vel_dynamics  # type: ignore[union-attr]

    thr_cmd = float(pa[env_idx, 0])
    thr_act = float(td._y[env_idx])
    av_cmd = pa[env_idx, 1:4].cpu().tolist()
    av_act = avd._ang_vel[env_idx].cpu().tolist()

    step = float(len(self._t))
    self._t.append(step)
    self._thr_cmd.append(thr_cmd)
    self._thr_act.append(thr_act)
    for i in range(3):
      self._av_cmd[i].append(av_cmd[i])
      self._av_act[i].append(av_act[i])

  def _show(self) -> None:
    self._visible = True
    if self._fig is None:
      self._create_figure()
    if self._fig is not None:
      self._fig.show()

  def _hide(self) -> None:
    self._visible = False
    if self._fig is not None:
      try:
        manager = self._fig.canvas.manager
        if manager is not None:
          manager.window.withdraw()  # type: ignore[union-attr]
      except Exception:
        plt.close(self._fig)
        self._fig = None

  def _create_figure(self) -> None:
    if self._fig is not None:
      return
    self._fig, self._axes = plt.subplots(
      4, 1, sharex=True, figsize=(10, 7), num="Action-Response"
    )
    labels = [
      "Thrust [N]",
      "Ang Vel X [rad/s]",
      "Ang Vel Y [rad/s]",
      "Ang Vel Z [rad/s]",
    ]
    for i, ax in enumerate(self._axes):
      (l_c,) = ax.plot([], [], "--", color=_CMD_COLOR, lw=1.0, label="cmd")
      (l_a,) = ax.plot([], [], "-", color=_ACT_COLOR, lw=1.0, label="act")
      self._ln_cmd.append(l_c)
      self._ln_act.append(l_a)
      ax.set_ylabel(labels[i], fontsize=8)
      ax.legend(loc="upper right", fontsize=7)
      ax.grid(True, alpha=0.3)
    self._axes[-1].set_xlabel("Step")
    self._fig.tight_layout(pad=0.5)

    def _on_close(_event):
      self._visible = False

    self._fig.canvas.mpl_connect("close_event", _on_close)

  def _redraw(self) -> None:
    if self._fig is None:
      return
    n = len(self._t)
    if n < 2:
      return

    t_arr = np.arange(-n + 1.0, 1.0)
    cmd_sets = (self._thr_cmd, self._av_cmd[0], self._av_cmd[1], self._av_cmd[2])
    act_sets = (self._thr_act, self._av_act[0], self._av_act[1], self._av_act[2])

    for i in range(4):
      self._ln_cmd[i].set_data(t_arr, list(cmd_sets[i]))
      self._ln_act[i].set_data(t_arr, list(act_sets[i]))
      self._axes[i].relim()
      self._axes[i].autoscale_view()

    self._fig.canvas.draw_idle()
    self._fig.canvas.flush_events()
