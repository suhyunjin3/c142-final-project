"""
modules/xenobots/__init__.py

Registers all xenobots tools with the FastMCP server instance.
Mirrors the pattern used by seq_basics.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .tools.xenobot_sim import XenobotSimulator

_sim = XenobotSimulator()


def register(mcp: FastMCP) -> None:
    """Attach all xenobots tools to *mcp*."""

    @mcp.tool(
        name="simulate_xenobot",
        description=(
            "Simulate the 2-D locomotion of a Xenobot given a list of cell "
            "coordinates and types ('passive', 'motor_x', 'motor_y'). "
            "Returns a base64-encoded PNG path visualisation plus kinematic "
            "statistics (net displacement, trajectory, cell counts)."
        ),
    )
    def simulate_xenobot(cells: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Parameters
        ----------
        cells : list of {"x": float, "y": float, "type": str}
            Body layout of the xenobot.  ``type`` must be one of
            ``'passive'``, ``'motor_x'``, or ``'motor_y'``.
        """
        if not _sim._initialized:
            _sim.initiate()
        return _sim.run(cells)
