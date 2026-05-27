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

# 4. Apply modified robosuite asset files (gripper + nut geometry)
python apply_patches.py
```

> `imageio[ffmpeg]` pulls in the FFmpeg backend for MP4 recording. If pip can't find `imageio[ffmpeg]` as a single token, install it separately:
> ```bash
> pip install imageio
> pip install imageio[ffmpeg]
> ```

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
