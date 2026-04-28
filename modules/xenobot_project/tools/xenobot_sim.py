"""
xenobot_sim.py — Xenobot Kinematics Simulator
BioE 234 Final Project

Simulates the emergent locomotion of a 2-D Xenobot constructed from
passive (skin/epidermis-derived) cells and motor (cardiac-muscle-derived)
cells, following the C6-Multiplatform model: data (cell layout) is
decoupled from simulation logic.

Cell types
----------
passive   : structural scaffold; contributes no net displacement force
motor_x   : cardiac cell driving force along the +X axis
motor_y   : cardiac cell driving force along the +Y axis

Kinematics model
----------------
Each motor cell adds a unit force vector to an accumulated net-force
vector.  The bot then "steps" in the net-force direction at each
simulated time-step; passive cells add a small isotropic drag that
scales the step size.  After N_STEPS the cumulative trajectory is
recorded and visualised as a 2-D path PNG.

This is an intentionally simplified Newtonian integration — suitable
for an undergraduate demonstration — rather than a full fluid-dynamics
or FEM simulation.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STEPS: int = 50          # number of integration steps
DT: float = 0.1            # time-step size (arbitrary units)
DRAG_COEFF: float = 0.05   # drag contribution per passive cell


class XenobotSimulator:
    """
    Simulates the 2-D kinematics of a Xenobot given a list of cell
    coordinates and type labels.

    Attributes
    ----------
    _initialized : bool
        True after `initiate()` has been called.
    """

    def __init__(self) -> None:
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # C9 lifecycle methods
    # ------------------------------------------------------------------

    def initiate(self) -> None:
        """Prepare the simulator (seed RNG, validate NumPy availability)."""
        np.random.seed(42)          # reproducible noise
        self._initialized = True

    def run(self, cells: list[dict]) -> dict:
        """
        Simulate xenobot locomotion and return trajectory data + PNG plot.

        Parameters
        ----------
        cells : list of dict
            Each dict must have:
              ``x``   (float) — cell x-coordinate in the body frame
              ``y``   (float) — cell y-coordinate in the body frame
              ``type``(str)   — one of ``'passive'``, ``'motor_x'``, ``'motor_y'``

        Returns
        -------
        dict
            ``content`` : list containing one image/png item and one text item::

                [
                    {"type": "image/png",
                     "data": "<base64-encoded PNG>",
                     "description": "Simulated Xenobot path"},
                    {"type": "text/plain",
                     "data": "net_displacement=...; steps=...; ..."}
                ]

            ``net_displacement`` : float — total Euclidean displacement (au)
            ``trajectory``       : list of [x, y] waypoints
            ``cell_summary``     : dict with counts per cell type
        """
        if not self._initialized:
            self.initiate()

        # ---- 1. Parse & validate cells -----------------------------------
        cell_array = _parse_cells(cells)

        # ---- 2. Compute net force vector ---------------------------------
        force = _compute_force(cell_array)

        # ---- 3. Compute drag from passive cells --------------------------
        n_passive = int(np.sum(cell_array[:, 2] == 0))
        drag = 1.0 / (1.0 + DRAG_COEFF * n_passive)   # scalar in (0, 1]

        # ---- 4. Integrate trajectory -------------------------------------
        trajectory = _integrate(force, drag)           # (N_STEPS+1, 2)

        # ---- 5. Compute summary statistics -------------------------------
        net_disp = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
        cell_summary = {
            "passive": int(np.sum(cell_array[:, 2] == 0)),
            "motor_x": int(np.sum(cell_array[:, 2] == 1)),
            "motor_y": int(np.sum(cell_array[:, 2] == 2)),
        }

        # ---- 6. Render path plot -----------------------------------------
        png_b64 = _render_path(trajectory, cell_array, cell_summary, net_disp)

        # ---- 7. Build C9-compatible result dict --------------------------
        summary_text = (
            f"net_displacement={net_disp:.4f} au; "
            f"steps={N_STEPS}; dt={DT}; "
            f"passive={cell_summary['passive']}; "
            f"motor_x={cell_summary['motor_x']}; "
            f"motor_y={cell_summary['motor_y']}"
        )

        return {
            "content": [
                {
                    "type": "image/png",
                    "data": png_b64,
                    "description": "Simulated Xenobot locomotion path",
                },
                {
                    "type": "text/plain",
                    "data": summary_text,
                },
            ],
            "net_displacement": net_disp,
            "trajectory": trajectory.tolist(),
            "cell_summary": cell_summary,
        }


# ---------------------------------------------------------------------------
# Internal helpers (module-private)
# ---------------------------------------------------------------------------

# Cell-type encoding
_TYPE_MAP: dict[str, int] = {"passive": 0, "motor_x": 1, "motor_y": 2}
_TYPE_COLORS: dict[str, str] = {
    "passive": "#5b8dd9",
    "motor_x": "#e85d5d",
    "motor_y": "#59c26e",
}


def _parse_cells(cells: list[dict]) -> np.ndarray:
    """
    Convert raw cell dicts to a float array of shape (N, 3).
    Columns: x, y, type_code
    Raises ValueError on invalid input.
    """
    if not cells:
        raise ValueError("cells list must not be empty")

    rows: list[list[float]] = []
    for i, cell in enumerate(cells):
        try:
            x = float(cell["x"])
            y = float(cell["y"])
            ctype = str(cell["type"]).lower().strip()
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Cell {i} missing required key 'x', 'y', or 'type': {exc}"
            ) from exc

        if ctype not in _TYPE_MAP:
            raise ValueError(
                f"Cell {i} has unknown type '{ctype}'. "
                f"Must be one of {list(_TYPE_MAP)}"
            )
        rows.append([x, y, float(_TYPE_MAP[ctype])])

    return np.array(rows, dtype=float)


def _compute_force(cell_array: np.ndarray) -> np.ndarray:
    """
    Sum unit force vectors contributed by motor cells.
    Returns a 2-D force vector [fx, fy].
    """
    motor_x_mask = cell_array[:, 2] == _TYPE_MAP["motor_x"]
    motor_y_mask = cell_array[:, 2] == _TYPE_MAP["motor_y"]

    fx = float(np.sum(motor_x_mask))
    fy = float(np.sum(motor_y_mask))
    force = np.array([fx, fy])

    # Normalise if non-zero so step length is controlled by DT & drag alone
    magnitude = np.linalg.norm(force)
    if magnitude > 1e-9:
        force = force / magnitude

    return force


def _integrate(force: np.ndarray, drag: float) -> np.ndarray:
    """
    Simple Euler integration of constant force + small Gaussian noise.
    Returns trajectory array of shape (N_STEPS+1, 2).
    """
    pos = np.zeros(2)
    path = [pos.copy()]

    noise_scale = 0.02          # small Brownian perturbation

    for _ in range(N_STEPS):
        noise = np.random.normal(0, noise_scale, size=2)
        velocity = force * drag + noise
        pos = pos + velocity * DT
        path.append(pos.copy())

    return np.array(path)


def _render_path(
    trajectory: np.ndarray,
    cell_array: np.ndarray,
    cell_summary: dict,
    net_disp: float,
) -> str:
    """
    Render a two-panel matplotlib figure:
      Left  — Xenobot body layout (cell positions coloured by type)
      Right — Simulated locomotion path (start → end)
    Returns a base64-encoded PNG string.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#0f1117")

    _style_ax(axes[0])
    _style_ax(axes[1])

    # --- Left panel: body layout ------------------------------------------
    type_labels = {0: "passive", 1: "motor_x", 2: "motor_y"}
    plotted: set[str] = set()
    for row in cell_array:
        cx, cy, tc = row[0], row[1], int(row[2])
        label_str = type_labels[tc]
        color = _TYPE_COLORS[label_str]
        lbl = label_str if label_str not in plotted else "_nolegend_"
        plotted.add(label_str)
        axes[0].scatter(
            cx, cy,
            s=220, c=color, edgecolors="white", linewidths=0.6,
            zorder=3, label=lbl,
        )

    axes[0].set_title("Xenobot Body Layout", color="white", fontsize=11, pad=8)
    axes[0].set_xlabel("x (body frame, au)", color="#aaaaaa", fontsize=9)
    axes[0].set_ylabel("y (body frame, au)", color="#aaaaaa", fontsize=9)
    legend = axes[0].legend(
        loc="upper right", fontsize=8,
        facecolor="#1e2130", edgecolor="#444", labelcolor="white",
    )

    # --- Right panel: trajectory ------------------------------------------
    xs, ys = trajectory[:, 0], trajectory[:, 1]

    # Colour-map path by time (cool-to-warm)
    n = len(xs) - 1
    cmap = plt.get_cmap("plasma")
    for i in range(n):
        t_frac = i / max(n - 1, 1)
        axes[1].plot(
            xs[i:i+2], ys[i:i+2],
            color=cmap(t_frac), linewidth=1.8, solid_capstyle="round",
        )

    # Start / end markers
    axes[1].scatter(xs[0],  ys[0],  s=80, c="#00e5ff", zorder=5, label="Start")
    axes[1].scatter(xs[-1], ys[-1], s=120, c="#ff6b35", marker="*", zorder=5, label="End")

    axes[1].set_title(
        f"Simulated Path  |  Δ = {net_disp:.3f} au",
        color="white", fontsize=11, pad=8,
    )
    axes[1].set_xlabel("x displacement (au)", color="#aaaaaa", fontsize=9)
    axes[1].set_ylabel("y displacement (au)", color="#aaaaaa", fontsize=9)
    axes[1].legend(
        loc="upper left", fontsize=8,
        facecolor="#1e2130", edgecolor="#444", labelcolor="white",
    )

    # Equal aspect on trajectory panel so circles are circles
    axes[1].set_aspect("equal", "datalim")

    fig.suptitle(
        "Xenobot Kinematics Simulator — BioE 234",
        color="#cccccc", fontsize=13, y=1.01,
    )
    plt.tight_layout()

    # Encode to base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _style_ax(ax: plt.Axes) -> None:
    """Apply dark-theme styling to a single Axes object."""
    ax.set_facecolor("#1a1d27")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
