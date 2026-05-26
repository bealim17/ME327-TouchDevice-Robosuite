"""
Haptic Calibration Demo
=======================
Pure MuJoCo scene — no robosuite — to verify position mapping and force direction.

Scene:
  - Blue sphere  = Touch stylus, moves in real-time
  - Red cube     = fixed object to touch and feel
  - Colored axes = X (red/right), Y (green/up), Z (blue/forward)

What to verify:
  1. Move stylus RIGHT  → sphere moves RIGHT on screen
  2. Move stylus UP     → sphere moves UP on screen
  3. Move stylus AWAY   → sphere moves into screen (away from camera)
  4. Push sphere into cube from any side → feel pushback in OPPOSITE direction

Touch natural axes used directly (no remapping):
  Touch X+ = your right  → Scene X+
  Touch Y+ = up          → Scene Y+
  Touch Z+ = toward you  → Scene Z+ (out of screen)
  Touch Z- = away        → Scene Z- (into screen)

Ctrl+C to quit.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
from dataclasses import dataclass, field

import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice


# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────

# Touch workspace center (mm) — from the calibration run
TOUCH_CENTER = np.array([-34.5, 43.9, -12.9])

# Scene scale: mm → meters (±200mm Touch = ±0.20m scene)
SCENE_SCALE = 0.001

# Force mode: "mj_contact" (mj_contactForce) or "penetration" (spring-damper)
FORCE_MODE  = "penetration"

# Force feedback — shared
MAX_FORCE_N = 2.5      # safety clip (Touch continuous limit ~3.3N)
FORCE_ALPHA = 0.2      # low-pass filter
FORCE_DAMP  = 0.0     # velocity damping on stylus (haptic callback)

# mj_contact mode
FORCE_SCALE = 0.05      # scale raw MuJoCo forces

# Penetration mode — spring-damper
STIFFNESS   = 800.0    # N/m — how hard the surface feels (tune up/down)
DAMPING     = 10.0     # N·s/m — damps oscillation at contact boundary


# ─────────────────────────────────────────────────
# MUJOCO SCENE XML
# ─────────────────────────────────────────────────

XML = """
<mujoco model="haptic_calibration">
  <option gravity="0 0 0" timestep="0.001"
          cone="elliptic" noslip_iterations="3"/>

  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.4 0.4 0.4"/>
    <scale contactwidth="0.01" contactheight="0.05" forcewidth="0.03"/>
    <map force="0.005"/>
  </visual>

  <default>
    <geom solimp="0.999 0.9999 0.0001" solref="0.002 0.5"/>
  </default>

  <worldbody>
    <light pos="0.5 1 1.5"  diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="false"/>
    <light pos="-0.5 1 1.5" diffuse="0.4 0.4 0.4" castshadow="false"/>

    <!-- Axis reference lines: red=X, green=Y, blue=Z -->
    <geom type="cylinder" fromto="-0.25 0 0  0.25 0 0"
          size="0.002" rgba="1 0.2 0.2 0.6" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 -0.25 0  0 0.25 0"
          size="0.002" rgba="0.2 1 0.2 0.6" contype="0" conaffinity="0"/>
    <geom type="cylinder" fromto="0 0 -0.25  0 0 0.25"
          size="0.002" rgba="0.2 0.2 1 0.6" contype="0" conaffinity="0"/>

    <!-- Small sphere at origin for reference -->
    <geom type="sphere" size="0.005" rgba="1 1 1 0.8"
          contype="0" conaffinity="0"/>

    <!-- Fixed red cube to touch — half size -->
    <body name="cube" pos="0 0 0">
      <geom name="cube_geom" type="box" size="0.03 0.03 0.03"
            rgba="0.9 0.15 0.15 1" contype="1" conaffinity="1"/>
    </body>

    <!-- Blue sphere = stylus proxy, free joint controlled via qpos -->
    <body name="stylus" pos="0.2 0 0">
      <freejoint name="stylus_joint"/>
      <geom name="stylus_geom" type="sphere" size="0.022"
            rgba="0.15 0.45 1.0 0.95" contype="1" conaffinity="1"
            mass="0.001"/>
    </body>

  </worldbody>
</mujoco>
"""


# ─────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────

@dataclass
class SharedState:
    stylus_pos_mm:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    stylus_vel_mm_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    contact_force_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    running:         bool = True

shared = SharedState()
lock   = threading.Lock()


# ─────────────────────────────────────────────────
# HAPTIC CALLBACK — 1kHz
# ─────────────────────────────────────────────────

_prev_pos  = np.zeros(3)
_prev_time = [time.perf_counter()]
_vel_filter = np.zeros(3)
VEL_ALPHA   = 0.05   # low-pass filter coefficient (lower = smoother, try 0.02-0.1)

@hd_callback
def haptic_callback():
    global _prev_pos

    handle = hd.get_current_device()
    hd.begin_frame(handle)

    # Position
    raw = hd._get_doublev(hd.HD_CURRENT_POSITION, hd.HDdouble * 3)
    pos = np.array(list(raw))

    # Velocity (mm/s) — low-pass filtered to reduce noise at high loop rate
    now = time.perf_counter()
    dt  = now - _prev_time[0]
    raw_vel = (pos - _prev_pos) / dt if dt > 1e-6 else np.zeros(3)
    _vel_filter[:] = VEL_ALPHA * raw_vel + (1 - VEL_ALPHA) * _vel_filter
    vel = _vel_filter.copy()
    _prev_pos     = pos.copy()
    _prev_time[0] = now

    # Read latest force from sim
    with lock:
        shared.stylus_pos_mm[:]   = pos
        shared.stylus_vel_mm_s[:] = vel
        force_cmd = shared.contact_force_N.copy()

    # Add velocity damping (makes contact feel less buzzy)
    damping     = -FORCE_DAMP * vel / 1000.0   # convert mm/s → m/s scale
    total_force = np.clip(force_cmd + damping, -MAX_FORCE_N, MAX_FORCE_N)

    hd.set_force(list(total_force))
    hd.end_frame(handle)


# ─────────────────────────────────────────────────
# POSITION MAPPING
# Touch natural axes map 1:1 to scene axes — no remapping needed.
# If sphere moves wrong direction, flip the sign on that axis below.
# ─────────────────────────────────────────────────

def touch_to_scene(pos_mm: np.ndarray) -> np.ndarray:
    centered = pos_mm - TOUCH_CENTER
    # Camera at azimuth=-90 (camera on -Y axis, looking toward +Y):
    #   MuJoCo X+ = screen right
    #   MuJoCo Z+ = screen up
    #   MuJoCo Y+ = out of screen (toward camera = closer)
    #   MuJoCo Y- = into screen (away from camera = further)
    #
    # Touch → MuJoCo:
    #   Touch X+ (right)  → MuJoCo X+  (screen right)
    #   Touch Y+ (up)     → MuJoCo Z+  (screen up)
    #   Touch Z+ (toward) → MuJoCo Y-  (toward you = Y- in this camera setup)
    scene = np.array([
         centered[0],   # Touch X  → MuJoCo  X (right)
        -centered[2],   # Touch Z  → MuJoCo -Y (toward you)
         centered[1],   # Touch Y  → MuJoCo  Z (up)
    ]) * SCENE_SCALE
    return scene


# ─────────────────────────────────────────────────
# CONTACT FORCE EXTRACTION
# ─────────────────────────────────────────────────

_filtered_force = np.zeros(3)

def get_contact_force_mj(model, data, stylus_body_id, stylus_geom_id):
    """Original mj_contactForce approach — physics forces, negated."""
    global _filtered_force
    net_force = np.zeros(3)
    force_buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 != stylus_geom_id and g2 != stylus_geom_id:
            continue
        mujoco.mj_contactForce(model, data, i, force_buf)
        contact_frame = c.frame.reshape(3, 3)
        f_world = contact_frame.T @ force_buf[:3]
        if g2 == stylus_geom_id:
            f_world = -f_world
        net_force += f_world
    _filtered_force = FORCE_ALPHA * net_force + (1 - FORCE_ALPHA) * _filtered_force
    return _filtered_force.copy()


def get_contact_force_penetration(model, data, stylus_geom_id, stylus_vel_mm_s):
    """
    Penetration-based spring-damper force.
    F = k * depth * normal + b * (vel · normal) * normal

    contact.dist < 0 means penetrating (negative = deeper)
    contact.frame[0:3] = contact normal pointing from geom2 to geom1

    This gives direct control over stiffness and damping,
    independent of MuJoCo's internal physics solver.
    """
    global _filtered_force
    net_force = np.zeros(3)
    stylus_vel_m_s = stylus_vel_mm_s / 1000.0  # mm/s → m/s

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 != stylus_geom_id and g2 != stylus_geom_id:
            continue

        # Penetration depth — contact.dist is negative when penetrating
        depth = -c.dist  # positive = how far we've penetrated
        if depth <= 0:
            continue     # not actually penetrating, skip

        # Contact normal in world frame (points from geom1 → geom2)
        normal = c.frame[:3].copy()

        # If stylus is geom1, normal points TOWARD geom2 (into surface)
        # We want normal pointing AWAY from surface (pushback direction)
        # So flip when stylus is geom1, keep when stylus is geom2
        if g1 == stylus_geom_id:
            normal = -normal

        # Stiffness term: pushes stylus out of surface
        f_stiffness = STIFFNESS * depth * normal

        # Damping term: damps velocity along normal direction
        vel_along_normal = np.dot(stylus_vel_m_s, normal)
        f_damping = -DAMPING * vel_along_normal * normal

        net_force += f_stiffness + f_damping

    _filtered_force = FORCE_ALPHA * net_force + (1 - FORCE_ALPHA) * _filtered_force
    return _filtered_force.copy()


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  Haptic Calibration Demo")
    print("=" * 52)

    # Build model
    model = mujoco.MjModel.from_xml_string(XML)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Get IDs
    stylus_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "stylus")
    stylus_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,  "stylus_geom")
    stylus_jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "stylus_joint")
    stylus_qpos_adr = model.jnt_qposadr[stylus_jnt_id]  # start of qpos for this joint

    print(f"  stylus body id : {stylus_body_id}")
    print(f"  stylus geom id : {stylus_geom_id}")
    print(f"  stylus qpos adr: {stylus_qpos_adr}")

    # Init Touch device
    print("\nInitializing Touch device...")
    device = HapticDevice(callback=haptic_callback, scheduler_type="async")
    hd.enable_force()
    print("Touch device ready.")

    print("\nVerification checklist:")
    print("  [ ] Move RIGHT  → sphere goes right")
    print("  [ ] Move UP     → sphere goes up")
    print("  [ ] Move AWAY   → sphere moves into screen")
    print("  [ ] Touch cube  → feel pushback away from cube")
    print("\nCtrl+C to quit.\n")

    step = 0
    loop_times = []
    prev_in_contact = False

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Camera at azimuth=90:
        # MuJoCo X+ = screen right,  Touch X+ = right   ✓
        # MuJoCo Z+ = screen up,     Touch Y+ = up       ✓
        # MuJoCo Y- = toward viewer, Touch Z+ = toward   ✓
        viewer.cam.azimuth   = 90
        viewer.cam.elevation = 0
        viewer.cam.distance  = 0.7
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]

        # Show contact forces as arrows in viewer
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        try:
            while viewer.is_running() and shared.running:
                t0 = time.perf_counter()

                # Read stylus position
                with lock:
                    stylus_mm = shared.stylus_pos_mm.copy()

                # Map to scene coordinates
                scene_pos = touch_to_scene(stylus_mm)

                # Drive sphere to stylus position via qpos (freejoint: pos + quat)
                data.qpos[stylus_qpos_adr:stylus_qpos_adr+3] = scene_pos
                data.qpos[stylus_qpos_adr+3:stylus_qpos_adr+7] = [1, 0, 0, 0]
                data.qvel[model.jnt_dofadr[stylus_jnt_id]:
                          model.jnt_dofadr[stylus_jnt_id]+6] = 0

                # Step physics
                mujoco.mj_step(model, data)

                # Override position again after step (physics may have moved it)
                data.qpos[stylus_qpos_adr:stylus_qpos_adr+3] = scene_pos
                data.qvel[model.jnt_dofadr[stylus_jnt_id]:
                          model.jnt_dofadr[stylus_jnt_id]+6] = 0

                # Forward pass to recompute cfrc_ext with new position
                mujoco.mj_forward(model, data)

                # Get contact forces
                with lock:
                    stylus_vel = shared.stylus_vel_mm_s.copy()

                if FORCE_MODE == "penetration":
                    raw_force = get_contact_force_penetration(
                        model, data, stylus_geom_id, stylus_vel)
                    # Penetration mode: force already points away from surface
                    # Just remap axes, no negation needed
                    mujoco_fx, mujoco_fy, mujoco_fz = raw_force
                    touch_force = np.array([
                         mujoco_fx,
                         mujoco_fz,
                        -mujoco_fy,
                    ])
                    scaled_force = touch_force  # stiffness already in N

                else:  # "mj_contact"
                    raw_force = get_contact_force_mj(
                        model, data, stylus_body_id, stylus_geom_id)
                    mujoco_fx, mujoco_fy, mujoco_fz = raw_force
                    touch_force = np.array([
                         mujoco_fx,
                         mujoco_fz,
                        -mujoco_fy,
                    ])
                    scaled_force = -touch_force * FORCE_SCALE

                scaled_force = np.clip(scaled_force, -MAX_FORCE_N, MAX_FORCE_N)

                in_contact = data.ncon > 0

                # Write to shared state for haptic callback
                with lock:
                    shared.contact_force_N[:] = scaled_force

                # Contact boundary diagnostics
                if in_contact:
                    depths = [-data.contact[i].dist for i in range(data.ncon)
                              if (data.contact[i].geom1 == stylus_geom_id
                              or data.contact[i].geom2 == stylus_geom_id)
                              and -data.contact[i].dist > 0]
                    max_depth = max(depths) if depths else 0
                    force_mag = np.linalg.norm(scaled_force)
                    print(f"  depth={max_depth*1000:.2f}mm | "
                          f"force={scaled_force.round(3)} N | "
                          f"mag={force_mag:.3f} N", end="\r")

                if not prev_in_contact and in_contact:
                    print(f"\n  >>> CONTACT START: force={scaled_force.round(3)} N")
                elif prev_in_contact and not in_contact:
                    print(f"\n  >>> CONTACT END")

                prev_in_contact = in_contact

                viewer.sync()

                # Measure actual loop rate
                loop_times.append(time.perf_counter() - t0)
                if len(loop_times) == 200:
                    avg_hz = 1.0 / np.mean(loop_times)
                    print(f"  Loop rate: {avg_hz:.1f} Hz  "
                          f"(avg {np.mean(loop_times)*1000:.2f} ms/loop)")
                    loop_times.clear()

                # Pace to timestep
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, model.opt.timestep - elapsed))

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            shared.running = False
            with lock:
                shared.contact_force_N[:] = np.zeros(3)
            time.sleep(0.2)
            device.close()
            print("Done.")