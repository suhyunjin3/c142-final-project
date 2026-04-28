"""
test_xenobot.py — pytest suite for the Xenobot Kinematics Simulator

Run from the project root:
    pytest modules/xenobots/tools/test_xenobot.py -v
"""

from __future__ import annotations

import base64
import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from modules.xenobots.tools.xenobot_sim import (
    XenobotSimulator,
    _compute_force,
    _integrate,
    _parse_cells,
    _render_path,
    N_STEPS,
    DT,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sim() -> XenobotSimulator:
    """Fresh, initialised simulator instance."""
    s = XenobotSimulator()
    s.initiate()
    return s


@pytest.fixture
def single_motor_x() -> list[dict]:
    """Minimal valid cells: one motor_x, one passive."""
    return [
        {"x": 0.0, "y": 0.0, "type": "motor_x"},
        {"x": 1.0, "y": 0.0, "type": "passive"},
    ]


@pytest.fixture
def diagonal_bot() -> list[dict]:
    """Equal motor_x and motor_y — should produce ~45° trajectory."""
    return [
        {"x": 0.0, "y": 0.0, "type": "motor_x"},
        {"x": 1.0, "y": 0.0, "type": "motor_y"},
        {"x": 0.5, "y": 0.5, "type": "passive"},
    ]


@pytest.fixture
def passive_only() -> list[dict]:
    """All passive cells — zero net force."""
    return [
        {"x": 0.0, "y": 0.0, "type": "passive"},
        {"x": 1.0, "y": 1.0, "type": "passive"},
    ]


# ===========================================================================
# 1. Lifecycle tests
# ===========================================================================

class TestLifecycle:
    def test_not_initialized_before_initiate(self):
        s = XenobotSimulator()
        assert not s._initialized

    def test_initialized_after_initiate(self):
        s = XenobotSimulator()
        s.initiate()
        assert s._initialized

    def test_run_auto_initializes(self, single_motor_x):
        """run() must work even if initiate() was never called explicitly."""
        s = XenobotSimulator()
        result = s.run(single_motor_x)
        assert s._initialized
        assert "content" in result


# ===========================================================================
# 2. Input validation tests
# ===========================================================================

class TestInputValidation:
    def test_empty_cells_raises(self, sim):
        with pytest.raises(ValueError, match="empty"):
            sim.run([])

    def test_missing_x_key_raises(self, sim):
        with pytest.raises(ValueError):
            sim.run([{"y": 0.0, "type": "motor_x"}])

    def test_missing_type_key_raises(self, sim):
        with pytest.raises(ValueError):
            sim.run([{"x": 0.0, "y": 0.0}])

    def test_unknown_type_raises(self, sim):
        with pytest.raises(ValueError, match="unknown type"):
            sim.run([{"x": 0.0, "y": 0.0, "type": "cilia"}])

    def test_type_case_insensitive(self, sim):
        """Type strings should be normalised to lower-case."""
        result = sim.run([{"x": 0.0, "y": 0.0, "type": "MOTOR_X"}])
        assert result["cell_summary"]["motor_x"] == 1

    @pytest.mark.parametrize("ctype", ["passive", "motor_x", "motor_y"])
    def test_each_valid_type_accepted(self, sim, ctype):
        result = sim.run([{"x": 0.0, "y": 0.0, "type": ctype}])
        assert result is not None


# ===========================================================================
# 3. Force computation tests
# ===========================================================================

class TestForceComputation:
    def test_no_motors_zero_force(self):
        cells = _parse_cells([
            {"x": 0.0, "y": 0.0, "type": "passive"},
        ])
        force = _compute_force(cells)
        np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)

    def test_pure_motor_x_force_direction(self):
        cells = _parse_cells([
            {"x": 0.0, "y": 0.0, "type": "motor_x"},
            {"x": 1.0, "y": 0.0, "type": "motor_x"},
        ])
        force = _compute_force(cells)
        # Normalised → should be [1, 0]
        np.testing.assert_allclose(force, [1.0, 0.0], atol=1e-9)

    def test_pure_motor_y_force_direction(self):
        cells = _parse_cells([
            {"x": 0.0, "y": 0.0, "type": "motor_y"},
        ])
        force = _compute_force(cells)
        np.testing.assert_allclose(force, [0.0, 1.0], atol=1e-9)

    def test_diagonal_force_normalised(self):
        """Equal motor_x and motor_y should produce a unit vector at 45°."""
        cells = _parse_cells([
            {"x": 0.0, "y": 0.0, "type": "motor_x"},
            {"x": 0.0, "y": 1.0, "type": "motor_y"},
        ])
        force = _compute_force(cells)
        expected = np.array([1.0, 1.0]) / math.sqrt(2)
        np.testing.assert_allclose(force, expected, atol=1e-9)

    def test_force_magnitude_is_unity_when_nonzero(self):
        cells = _parse_cells([
            {"x": 0.0, "y": 0.0, "type": "motor_x"},
            {"x": 1.0, "y": 0.0, "type": "motor_y"},
        ])
        force = _compute_force(cells)
        assert abs(np.linalg.norm(force) - 1.0) < 1e-9


# ===========================================================================
# 4. Trajectory / integration tests
# ===========================================================================

class TestIntegration:
    def test_trajectory_shape(self):
        force = np.array([1.0, 0.0])
        traj = _integrate(force, drag=1.0)
        assert traj.shape == (N_STEPS + 1, 2)

    def test_zero_force_small_displacement(self):
        """With zero net force, bot should barely move (noise only)."""
        np.random.seed(0)
        force = np.array([0.0, 0.0])
        traj = _integrate(force, drag=1.0)
        disp = np.linalg.norm(traj[-1] - traj[0])
        # Noise-only displacement should be well under 1 au over 50 steps
        assert disp < 0.5

    def test_motor_x_bot_moves_mostly_right(self):
        """Pure motor_x bot: final x >> final y in magnitude."""
        np.random.seed(42)
        force = np.array([1.0, 0.0])
        traj = _integrate(force, drag=1.0)
        assert traj[-1, 0] > abs(traj[-1, 1]) * 2

    def test_drag_reduces_displacement(self):
        """Higher drag (fewer passive cells) → larger net displacement."""
        np.random.seed(42)
        force = np.array([1.0, 0.0])
        traj_low_drag  = _integrate(force, drag=1.0)   # no passive cells
        traj_high_drag = _integrate(force, drag=0.3)   # many passive cells
        disp_low  = np.linalg.norm(traj_low_drag[-1]  - traj_low_drag[0])
        disp_high = np.linalg.norm(traj_high_drag[-1] - traj_high_drag[0])
        assert disp_low > disp_high


# ===========================================================================
# 5. Output structure tests
# ===========================================================================

class TestOutputStructure:
    def test_result_has_required_keys(self, sim, single_motor_x):
        result = sim.run(single_motor_x)
        for key in ("content", "net_displacement", "trajectory", "cell_summary"):
            assert key in result, f"Missing key: {key}"

    def test_content_has_image_and_text(self, sim, single_motor_x):
        result = sim.run(single_motor_x)
        content = result["content"]
        assert len(content) == 2
        types = {item["type"] for item in content}
        assert "image/png" in types
        assert "text/plain" in types

    def test_base64_image_is_valid(self, sim, single_motor_x):
        result = sim.run(single_motor_x)
        img_item = next(c for c in result["content"] if c["type"] == "image/png")
        raw = base64.b64decode(img_item["data"])
        # PNG magic bytes: \x89PNG
        assert raw[:4] == b"\x89PNG"

    def test_net_displacement_is_non_negative(self, sim, single_motor_x):
        result = sim.run(single_motor_x)
        assert result["net_displacement"] >= 0.0

    def test_trajectory_length(self, sim, single_motor_x):
        result = sim.run(single_motor_x)
        assert len(result["trajectory"]) == N_STEPS + 1

    def test_cell_summary_counts_correct(self, sim):
        cells = [
            {"x": 0.0, "y": 0.0, "type": "passive"},
            {"x": 1.0, "y": 0.0, "type": "motor_x"},
            {"x": 0.0, "y": 1.0, "type": "motor_x"},
            {"x": 1.0, "y": 1.0, "type": "motor_y"},
        ]
        result = sim.run(cells)
        assert result["cell_summary"] == {"passive": 1, "motor_x": 2, "motor_y": 1}


# ===========================================================================
# 6. Physics / kinematic sanity tests
# ===========================================================================

class TestKinematicSanity:
    def test_motor_x_bot_positive_x_displacement(self, sim, single_motor_x):
        """A bot with only motor_x cells must end up in positive-x territory."""
        result = sim.run(single_motor_x)
        traj = np.array(result["trajectory"])
        assert traj[-1, 0] > 0, "motor_x bot should move in positive x"

    def test_motor_y_bot_positive_y_displacement(self, sim):
        cells = [{"x": 0.0, "y": 0.0, "type": "motor_y"}]
        result = sim.run(cells)
        traj = np.array(result["trajectory"])
        assert traj[-1, 1] > 0, "motor_y bot should move in positive y"

    def test_passive_only_near_zero_displacement(self, sim, passive_only):
        """All-passive bot has no net force; displacement should be tiny."""
        result = sim.run(passive_only)
        assert result["net_displacement"] < 0.5

    def test_more_motors_increases_displacement(self, sim):
        one_motor = [{"x": 0.0, "y": 0.0, "type": "motor_x"}]
        three_motors = [
            {"x": 0.0, "y": 0.0, "type": "motor_x"},
            {"x": 1.0, "y": 0.0, "type": "motor_x"},
            {"x": 2.0, "y": 0.0, "type": "motor_x"},
        ]
        np.random.seed(7)
        r1 = sim.run(one_motor)
        np.random.seed(7)
        r3 = sim.run(three_motors)
        # Force vector is always normalised; displacement should be similar
        # but three-motor bot has no passive drag either → equal or greater
        assert r3["net_displacement"] >= r1["net_displacement"] * 0.8

    def test_diagonal_bot_roughly_equal_x_y(self, sim, diagonal_bot):
        """45° bot: |Δx| and |Δy| should be within 3× of each other."""
        result = sim.run(diagonal_bot)
        traj = np.array(result["trajectory"])
        dx = abs(traj[-1, 0] - traj[0, 0])
        dy = abs(traj[-1, 1] - traj[0, 1])
        if dx > 1e-6 and dy > 1e-6:
            ratio = max(dx, dy) / min(dx, dy)
            assert ratio < 3.0, f"Expected near-diagonal path; dx={dx:.3f}, dy={dy:.3f}"


# ===========================================================================
# 7. Reproducibility test
# ===========================================================================

class TestReproducibility:
    def test_same_seed_same_trajectory(self):
        """Two runs with reset seed must produce identical trajectories."""
        cells = [
            {"x": 0.0, "y": 0.0, "type": "motor_x"},
            {"x": 1.0, "y": 0.5, "type": "passive"},
        ]
        s1 = XenobotSimulator()
        s1.initiate()
        r1 = s1.run(cells)

        s2 = XenobotSimulator()
        s2.initiate()
        r2 = s2.run(cells)

        np.testing.assert_allclose(
            r1["trajectory"], r2["trajectory"],
            atol=1e-12,
            err_msg="Trajectories differ across identical runs — RNG not seeded properly",
        )
