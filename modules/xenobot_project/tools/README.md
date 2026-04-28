# Xenobot Kinematics Simulator — BioE 234 Final Project

> **Module:** `modules/xenobots`
> **C9 Function ID:** `org.c9.function.robotics.xenobots.kinematics.v1`
> **Author:** BioE 234 Student Module

---

## Overview

This module simulates the emergent two-dimensional locomotion of a **Xenobot** — a programmable living organism built from *Xenopus laevis* embryonic cells — using a simplified Newtonian kinematics model.

You provide a spatial layout of cells (x, y positions and type labels), and the simulator integrates a trajectory, then returns a labelled PNG visualisation of the bot's path alongside quantitative kinematic metrics.

### What is a Xenobot?

Xenobots are the first class of "living machines": millimetre-scale organisms assembled *in silico* by an evolutionary algorithm (Kriegman et al., 2020) and then built by microsurgery from two cell types:

| Cell type | Biological origin | Role in this model |
|-----------|------------------|--------------------|
| `passive` | Epidermis (skin) | Structural scaffold; adds drag |
| `motor_x` | Cardiac muscle   | Contractile force along +x |
| `motor_y` | Cardiac muscle   | Contractile force along +y |

---

## Project Structure

```
modules/
└── xenobots/
    ├── __init__.py               ← MCP tool registration (register_all hook)
    └── tools/
        ├── __init__.py
        ├── xenobot_sim.py        ← Core simulation class (XenobotSimulator)
        ├── xenobot_sim.json      ← C9 JSON wrapper / function schema
        ├── prompts.json          ← 5 realistic Gemini prompt templates
        └── test_xenobot.py       ← pytest test suite (30+ assertions)
```

This layout mirrors the `seq_basics` pattern: every tool lives in a `tools/` subdirectory paired with its `.json` wrapper.

---

## Kinematics Model

1. **Force vector** — each `motor_x` cell contributes `[1, 0]` and each `motor_y` cell contributes `[0, 1]` to a cumulative force vector, which is then *normalised* to a unit vector so direction is determined by cell-type *ratio*, not count.

2. **Drag** — passive cells add structural mass. A scalar drag coefficient `drag = 1 / (1 + 0.05 × n_passive)` scales the step size at each integration step.

3. **Euler integration** — position is updated for `N_STEPS = 50` steps at `dt = 0.1` au per step:

   ```
   velocity  = force × drag + Gaussian_noise(σ=0.02)
   position += velocity × dt
   ```

4. **Visualisation** — a two-panel dark-theme PNG is generated:
   - **Left:** body layout coloured by cell type
   - **Right:** colour-mapped locomotion path (cool → warm with time)

---

## Installation

Install dependencies into your existing project environment:

```bash
pip install numpy matplotlib fastmcp
```

---

## Usage

### As an MCP Tool (via Gemini client)

Start the server and the Gemini chat client as usual:

```bash
python client_gemini.py
```

The tool `simulate_xenobot` will appear in the discovered tools list. Example prompt:

```
Simulate a xenobot with two motor_x cells at (0,0) and (1,0) and one passive
cell at (0.5, 1.0). How far does it travel?
```

### Direct Python API

```python
from modules.xenobots.tools.xenobot_sim import XenobotSimulator

sim = XenobotSimulator()
sim.initiate()

cells = [
    {"x": 0.0, "y": 0.0, "type": "motor_x"},
    {"x": 1.0, "y": 0.0, "type": "motor_x"},
    {"x": 0.5, "y": 1.0, "type": "passive"},
]

result = sim.run(cells)

print(f"Net displacement: {result['net_displacement']:.4f} au")
print(f"Cell summary: {result['cell_summary']}")

# Decode and save the PNG
import base64
png_bytes = base64.b64decode(result["content"][0]["data"])
with open("xenobot_path.png", "wb") as f:
    f.write(png_bytes)
```

### Running Tests

```bash
# From the project root
pytest modules/xenobots/tools/test_xenobot.py -v
```

Expected output: **30+ tests passing** across lifecycle, validation, force computation, trajectory integration, output structure, physics sanity, and reproducibility checks.

---

## C9 JSON Schema Compliance

The file `xenobot_sim.json` strictly follows the schema defined in `Function_Development_Specification.md`:

| Required field | Value |
|----------------|-------|
| `id` | `org.c9.function.robotics.xenobots.kinematics.v1` |
| `name` | Xenobot Kinematics Simulator |
| `type` | `"function"` |
| `inputs` | `cells` array with x, y, type per cell |
| `outputs` | dict with `content`, `net_displacement`, `trajectory`, `cell_summary` |
| `examples` | 2 worked examples with inputs and expected outputs |
| `execution_details` | `language`, `source`, `initialization`, `execution`, `mcp_name`, `seq_params` |

---

## Design Decisions

**Why normalise the force vector?** The biologically meaningful design variable is the *arrangement and ratio* of cell types, not how many motor cells are present in absolute terms. Normalisation keeps step size consistent and focuses experiments on directional design — consistent with how Kriegman et al. parametrised their evolutionary search.

**Why decouple `XenobotSimulator` from the MCP registration?** Following the C6-Multiplatform principle: the simulator class is a pure Python "Verb" that can be unit-tested, imported standalone, or called from any host (Flask, FastAPI, Jupyter) without FastMCP present. The `__init__.py` adapter is the thin "glue" layer.

**Why Euler integration instead of Runge-Kutta?** This is an undergraduate demonstration module. The forces are constant and the noise term is small; Euler integration is exact for constant acceleration and the added complexity of RK4 would obscure the core biological intuition.

---

## References

- Kriegman, S. et al. (2020). *A scalable pipeline for designing reconfigurable organisms.* PNAS, 117(4), 1853–1859.
- Blackiston, D. et al. (2021). *A cellular platform for the development of synthetic living machines.* Science Robotics, 6(52).
