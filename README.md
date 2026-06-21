# ME327 Touch Device + Robosuite Haptic Teleoperation

Teleoperate a Panda robot arm in [robosuite](https://robosuite.ai/) using a **3D Systems Touch**. The Touch device maps to the robot's end-effector position and renders contact forces back to your hand.

Built for ME327 Project at Stanford.
Author: Bea Lim

---

## Hardware Requirements

- **3D Systems Touch** haptic device, connected via USB
- Windows 10/11 PC with the 3D Systems **OpenHaptics SDK** installed

> The Touch device runs its servo loop at 1 kHz via the OpenHaptics driver. This is a hard requirement — the software will not launch without the device connected and the driver running.

---

## Software Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.13 |
| OpenHaptics SDK | 3.5+ (from 3D Systems) |
| Git | any |

Install **OpenHaptics** from the [3D Systems developer portal](https://www.3dsystems.com/haptics-devices/openhaptics) before continuing. `pyOpenHaptics` is just a Python wrapper — it needs the underlying SDK DLLs on PATH.

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/bealim17/ME327-TouchDevice-Robosuite.git
cd ME327-TouchDevice-Robosuite

# 2. Create a Python 3.13 virtual environment
# If you installed Python from python.org (includes the Python Launcher):
py -3.13 -m venv .venv
# Otherwise, make sure Python 3.13 is on your PATH and use:
# python -m venv .venv

.venv\Scripts\activate        # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Upgrade numpy to 2.x (robosuite's mink sub-dependency declares numpy<2.0,
#    so install the rest first, then force-upgrade)
pip install numpy==2.4.6

# 5. Apply modified robosuite asset files (gripper + nut geometry)
python apply_patches.py
```

---

## Running

### Main teleoperation script (with video recording)
```bash
python robosuite_haptic_bridge.py
```

### No-recording variant (slightly lower overhead)
```bash
python robosuite_haptic_bridge_norecord.py
```

### Standalone MuJoCo calibration demo (no robosuite, no robot)
```bash
python haptic_calibration_demo.py
```
Use this first to verify axis mapping and force directions without loading the full robot environment.

### Controller Characterization (no haptic device required)
```bash
# Run all experiments + analysis in one shot
python experiments/osc_tests.py

# Run experiments only (skip analysis — set RUN_ANALYSIS = False in osc_tests.py)
python experiments/osc_tests.py

# Run analysis only on existing CSVs in experiments/data/
python experiments/analyze_osc_results.py
```

Results are saved to:
- `experiments/data/` — raw CSVs per trial + summary tables
- `experiments/figures/` — all plots

### Haptic Force Rendering Quality (no haptic device required)
```bash
python experiments/haptic_simhz_sweep.py
```
Scripted table-impact sweep across SIM_HZ = [10, 20, 50, 100]. Reconstructs the 1 kHz zero-order-hold haptic signal from sim data and compares staircase fidelity across rates.

---

## Controller Characterization (`experiments/osc_tests.py`)

Systematic open-loop characterization of the OSC position controller. All tests run scripted trajectories — no Touch device or human input required. See [Section 3.2 of the project wiki](https://charm.stanford.edu/ME327/2026-Group18#toc5) for full methodology and results.

### Experiments

**Step Response** — 15–20 cm step along X, Y, Z at t = 2 s, 10 s trial.
Extracts rise time, settling time (±2% band), overshoot, and steady-state RMS error per axis.
Key result: 200–250 ms rise time, ~12% overshoot on X/Y, ~6% on Z (gravity compensation reduces effective inertia).

**Frequency Response & Bandwidth** — Sinusoidal targets at 0.5, 1.0, 2.0, 3.0, 5.0 Hz (5 cm amplitude, Y-axis, 20 s each).
Computes amplitude gain, phase lag, and RMS tracking error per frequency. Estimates −3 dB bandwidth by interpolation.
Key result: bandwidth ≈ 1.5 Hz; 90° phase-lag crossover at ~1.5 Hz is the practical haptic stability limit — beyond this, reflected forces invert (assisting insertion, resisting withdrawal).

**Damping Ratio Sweep** — OSC damping ratio ζ ∈ {0.2, 0.5, 1.0, 2.0}, both step and 0.5 Hz sine inputs.
Step responses are indistinguishable (controller saturates at max effort during transients); sine tracking reveals ζ = 0.2 gives lowest phase lag (150 ms vs 180 ms) and lowest peak error.
Key result: ζ = 0.2 recommended.

**Simulation Rate Sweep** — SIM_HZ ∈ {10, 20, 50, 100} Hz, 0.5 Hz sine for 20 s.
Measures RMS position error and zero-order-hold force update latency per rate.
Key result: 10 → 100 Hz cuts RMS error by 41% (34 mm → 20 mm); diminishing returns above 50 Hz; 100 Hz chosen for 10 ms ZOH latency during impulsive contact.

**Cross-Axis Coupling** — Off-axis displacement during primary-axis step inputs, relative to pre-step baseline.
Key result: X–Z coupling dominates (33 mm transient X-drift during Z-step). All coupling is transient and decays within 2 s, but the 33 mm peak exceeds typical peg-hole clearance (27.5 mm radius) — slow approach (<0.1 m/s) is required.

### Enabling / Disabling Individual Tests

At the top of `__main__` in `experiments/osc_tests.py`:

```python
RUN_STEP_RESPONSE   = True
RUN_FREQUENCY_SWEEP = True
RUN_DAMPING_SWEEP   = True
RUN_KP_SWEEP        = True
RUN_SIMHZ_SWEEP     = True
RUN_ANALYSIS        = True   # set False to skip post-processing
```

Set any flag to `False` to skip that experiment or the analysis pass.

---

## Controls

| Input | Action |
|---|---|
| Move Touch stylus | Move robot end-effector |
| Touch Button 1 (bottom) | Toggle gripper open / close |
| `Z` key | Toggle force feedback on / off |
| `V` key | Toggle video recording on / off |
| `Spacebar` | Reset simulation |
| `Ctrl+C` | Quit (in Terminal) |

---

## Scene

The environment (`round_nut_only_env.py`) loads a Panda arm on a table with:

- **Round nut**: fixed at the table center, acts as the insertion target
- **Cylinder peg**: free-moving, randomly placed in a semicircle around the nut each reset

The task is to pick up and insert the cylinder into the nut using only haptic feedback. Object colors are intentionally similar (both grey) to encourage reliance on touch over vision.

---

## File Overview

| File | Purpose |
|---|---|
| `robosuite_haptic_bridge.py` | Main teleoperation loop with video recording |
| `robosuite_haptic_bridge_norecord.py` | Same, without recording (lower overhead) |
| `round_nut_only_env.py` | Custom robosuite environment (round nut + cylinder) |
| `osc_world_frame.json` | OSC_POSE controller config (world-frame delta position) |
| `haptic_calibration_demo.py` | Standalone MuJoCo scene for axis/force calibration |
| `check_controller_robo.py` | Quick sanity check for robosuite controller loading |
| `robosuite_patches/` | Modified robosuite asset files (gripper XML + STL, round-nut XML) |
| `apply_patches.py` | Copies `robosuite_patches/` files into your robosuite installation |

---

## Axis Mapping

The Touch device's coordinate system is remapped to MuJoCo world frame as confirmed in `haptic_calibration_demo.py`:

```
TOUCH TO WORLD / VIRTUAL ENV
Position:  Touch X  →  World X   (right / left)
           Touch Y  →  World Z   (up / down)
           Touch Z  →  World −Y  (depth, negated)

WORLD / VIRTUAL ENV TO TOUCH
Force:     World X  →  Touch X
           World Z  →  Touch Y
           World Y  →  Touch −Z  (negated — Newton's 3rd law)
```

---

## Tuning Parameters

Key parameters at the top of `robosuite_haptic_bridge.py`:

| Parameter | Default | Effect |
|---|---|---|
| `MAPPING_MODE` | `"absolute"` | `"absolute"` maps Touch workspace to robot workspace; `"relative"` uses velocity-based control |
| `ACTION_GAIN` | `10.0` | Scales end-effector speed |
| `MAX_FORCE_N` | `1.5` | Haptic force clamp (N) — keep below ~3 N for comfort/device safety |
| `FORCE_SCALE` | `0.01` | Scales MuJoCo contact forces to haptic output, value derived from calibration |
| `SIM_HZ` | `50` | Simulation / control frequency |

---

## Robosuite Patches

Three robosuite asset files were modified from stock and are stored in `robosuite_patches/`. `apply_patches.py` copies them into your environment automatically (step 4 of installation).

| Patched file | What changed |
|---|---|
| `models/assets/grippers/panda_gripper.xml` | Added `finger_joint1_tip` / `finger_joint2_tip` bodies with high-friction pad geoms (`friction="3.2 0.05 0.0001"`, `solref="0.01 0.5"`) for reliable grasp. Also declares the custom `finger_longer.stl` mesh asset. |
| `models/assets/grippers/meshes/panda_gripper/finger_longer.stl` | Custom STL mesh referenced by the patched gripper XML. |
| `models/assets/objects/round-nut.xml` | Geometry and friction parameters tuned for the insertion task. |

If you skip `apply_patches.py` the gripper will use stock robosuite friction values and the nut geometry will differ from what the teleoperation was tuned for.

---

## Dependencies

See `requirements.txt`. Key packages:

- [mujoco](https://github.com/google-deepmind/mujoco) 3.8.1 — physics simulation
- [robosuite](https://robosuite.ai/) 1.5.2 — robot environment framework
- [pyOpenHaptics](https://github.com/mikedeitz/pyOpenHaptics) 1.0.1 — Python bindings for OpenHaptics
- [pynput](https://github.com/moses-palmer/pynput) — keyboard listener
- [imageio](https://imageio.readthedocs.io/) + FFmpeg — MP4 recording
