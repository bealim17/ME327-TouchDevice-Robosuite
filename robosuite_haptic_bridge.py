"""
Robosuite Haptic Teleoperation Bridge
======================================
Controls Panda arm in robosuite using 3D Systems Touch device.

Confirmed axis mappings from haptic_calibration_demo.py:

  Position (Touch → MuJoCo world):
    Touch X → World X  (right/left)
    Touch Y → World Z  (up/down,  MuJoCo Z-up)
    Touch Z → World -Y (depth, negated)

  Force (MuJoCo world → Touch):
    MuJoCo X → Touch X   (same)
    MuJoCo Z → Touch Y   (up/down)
    MuJoCo Y → Touch -Z  (negated, inverse of position mapping)
    + negate final result (Newton's 3rd law — mj_contactForce gives
      force finger exerts ON object, we want force object exerts ON finger)

  Velocity: low-pass filtered at VEL_ALPHA=0.05 (~143Hz cutoff at 1kHz)
  Forces:   low-pass filtered at FORCE_ALPHA

Controls:
  Move Touch stylus         : move robot end effector
  Touch device Button 1     : toggle gripper open/close (bottom button)
  Z key                     : toggle force feedback on/off
  SPACEBAR                  : reset simulation
  Ctrl+C                    : quit
"""

import threading
import time
import numpy as np
from dataclasses import dataclass, field

import mujoco
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice

from pynput import keyboard


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

MAPPING_MODE = "absolute"   # "absolute" or "relative"

# Touch workspace bounds (mm) from calibration
TOUCH_X_RANGE = (-218.4, 149.5)
TOUCH_Y_RANGE = (-112.9, 200.7)
TOUCH_Z_RANGE = (-115.2,  89.3)

TOUCH_CENTER = np.array([
    (TOUCH_X_RANGE[0] + TOUCH_X_RANGE[1]) / 2,
    (TOUCH_Y_RANGE[0] + TOUCH_Y_RANGE[1]) / 2,
    (TOUCH_Z_RANGE[0] + TOUCH_Z_RANGE[1]) / 2,
])

# Robot workspace — center and half-range (m)
# Auto-updated at startup to match actual reset position
ROBOT_CENTER     = np.array([-0.087,  0.001,  1.022])  # from axis test reset pos
ROBOT_HALF_RANGE = np.array([ 0.45,   0.45,   0.4 ]) # position scaling

# Relative mode
RELATIVE_SPEED = 0.002     # m per mm displacement per step — feels responsive
DEADZONE_MM    = 5.0       # small deadzone
MAX_DISP_MM    = 80.0      # clamp displacement per axis (mm)

# Action
# BASIC controller OSC_POSE: output_max=0.05m/step, input scaled [-1,1]
# So action=1.0 → 0.05m movement per step at SIM_HZ=20 → 1m/s max
ACTION_GAIN = 5.0          # gentle — controller has ramp_ratio=0.2 built in

# Force feedback
FORCE_FEEDBACK_ENABLED = True   # True = render haptic forces, False = disable force feedback
FORCE_MODE    = "mj_contact"  # "mj_contact":mujoco contact forces / "penetration": for contact forces rendered via OpenHaptics
FORCE_BODY    = "gripper"        # "gripper": always gripper forces
                               # "object": always object forces
                               # "auto": gripper when open, object when closed

# General Tuning parameters for Force Feedback
MAX_FORCE_N   = 1.5        # Touch device force clip (N)
FORCE_ALPHA   = 1.0        # force low-pass filter (1.0 = no smoothing, 0.0 = maximum smoothing)

# Tuning parameters for mj_contact Force Feedback — adjust for desired feel - FORCE_MODE = "mj_contact" / default
FORCE_SCALE   = 0.01      # scale raw MuJoCo forces for Touch device (tune for desired feel)
FORCE_DAMP    = 0.0         # damping — only applied when in contact

# Tuning parameters for penetration Force Feedback mode (OpenHaptics spring-damper) - FORCE_MODE = "penetration"
STIFFNESS     = 150.0      # N/m — start low, increase if surface feels too soft
DAMPING       = 0.0        # N·s/m — damps oscillation at contact boundary

# Gripper geom names — all contact surfaces from robosuite diagnostic
FINGER_GEOMS = {
    'gripper0_right_finger1_collision',
    'gripper0_right_finger2_collision',
    'gripper0_right_finger1_pad_collision',
    'gripper0_right_finger2_pad_collision',
    'gripper0_right_hand_collision',
}

SIM_HZ = 50


# ─────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────

@dataclass
class SharedState:
    stylus_pos_mm:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    stylus_vel_mm_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    button1_pressed: bool = False
    contact_force_N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_closed:  bool = False
    last_button1:    bool = False
    running:         bool = True
    reset_requested: bool = False
    force_feedback_enabled: bool = FORCE_FEEDBACK_ENABLED

shared = SharedState()
lock   = threading.Lock()

# ─────────────────────────────────────────────────────────────
# KEYBOARD CALLBACK
# ─────────────────────────────────────────────────────────────

def on_key_press(key):
    try:
        if key == keyboard.Key.space:
            with lock:
                shared.reset_requested = True
            print("  [SPACEBAR] Reset requested")
        elif hasattr(key, 'char') and key.char == 'z':
            with lock:
                shared.force_feedback_enabled = not shared.force_feedback_enabled
                new_state = shared.force_feedback_enabled
            print(f"  [Z] Force feedback set to {new_state}")
    except AttributeError:
        pass


# ─────────────────────────────────────────────────────────────
# HAPTIC CALLBACK — 1kHz
# ─────────────────────────────────────────────────────────────

_prev_pos   = np.zeros(3)
_prev_time  = [time.perf_counter()]
_vel_filter = np.zeros(3)
VEL_ALPHA   = 0.05   # velocity low-pass filter (~143Hz cutoff at 1kHz)

@hd_callback
def haptic_callback():
    global _prev_pos

    handle = hd.get_current_device()
    hd.begin_frame(handle)

    raw = hd._get_doublev(hd.HD_CURRENT_POSITION, hd.HDdouble * 3)
    pos = np.array(list(raw))

    raw_btn = hd._get_integerv(hd.HD_CURRENT_BUTTONS, hd.HDint * 1)
    button1 = bool(list(raw_btn)[0] & hd.HD_DEVICE_BUTTON_1)

    now = time.perf_counter()
    dt  = now - _prev_time[0]
    raw_vel = (pos - _prev_pos) / dt if dt > 1e-6 else np.zeros(3)
    _vel_filter[:] = VEL_ALPHA * raw_vel + (1 - VEL_ALPHA) * _vel_filter
    vel = _vel_filter.copy()
    _prev_pos     = pos.copy()
    _prev_time[0] = now

    with lock:
        shared.stylus_pos_mm[:]   = pos
        shared.stylus_vel_mm_s[:] = vel
        shared.button1_pressed    = button1
        force_cmd = shared.contact_force_N.copy() if shared.force_feedback_enabled else np.zeros(3)

    # Velocity damping — only applied when in contact, not in free space
    in_contact  = np.linalg.norm(force_cmd) > 1e-4
    damping     = -FORCE_DAMP * vel / 1000.0 if in_contact else np.zeros(3)
    total_force = np.clip(force_cmd + damping, -MAX_FORCE_N, MAX_FORCE_N)

    hd.set_force(list(total_force))
    hd.end_frame(handle)


# ─────────────────────────────────────────────────────────────
# POSITION MAPPING
# ─────────────────────────────────────────────────────────────

def touch_to_world(pos_mm: np.ndarray) -> np.ndarray:
    """
    Convert Touch position (mm) to robot world frame (m).
    Confirmed from haptic_calibration_demo.py:
      Touch X+ (right)  → World Y+ (right on screen)
      Touch Y+ (up)     → World Z+ (up, MuJoCo Z-up)
      Touch Z+ (toward) → World X+ (toward camera)
    """
    centered = pos_mm - TOUCH_CENTER
    return np.array([
         centered[2],   # Touch Z → World X  (toward camera = +X)
         centered[0],   # Touch X → World Y  (right on screen = +Y)
         centered[1],   # Touch Y → World Z  (up = +Z)
    ]) * 0.001          # mm → m


def compute_target_absolute(stylus_pos_mm: np.ndarray) -> np.ndarray:
    """Map full Touch workspace to robot workspace."""
    touch_range = np.array([
        TOUCH_X_RANGE[1] - TOUCH_X_RANGE[0],
        TOUCH_Z_RANGE[1] - TOUCH_Z_RANGE[0],   # Z range for Y axis
        TOUCH_Y_RANGE[1] - TOUCH_Y_RANGE[0],   # Y range for Z axis
    ])
    touch_min = np.array([TOUCH_X_RANGE[0], TOUCH_Z_RANGE[0], TOUCH_Y_RANGE[0]])

    world = touch_to_world(stylus_pos_mm)
    world_range = np.array([
        TOUCH_X_RANGE[1] - TOUCH_X_RANGE[0],
        TOUCH_Z_RANGE[1] - TOUCH_Z_RANGE[0],
        TOUCH_Y_RANGE[1] - TOUCH_Y_RANGE[0],
    ]) * 0.001

    # Normalize to [-1, 1] then scale to robot workspace
    norm = np.clip(world / (world_range / 2), -1, 1)
    return ROBOT_CENTER + norm * ROBOT_HALF_RANGE


def compute_target_relative(stylus_pos_mm: np.ndarray,
                             eef_pos: np.ndarray,
                             home_mm: np.ndarray) -> np.ndarray:
    displacement = stylus_pos_mm - home_mm

    # Deadzone
    for i in range(3):
        if abs(displacement[i]) < DEADZONE_MM:
            displacement[i] = 0.0

    # Clamp max displacement per axis to prevent large jumps
    displacement = np.clip(displacement, -MAX_DISP_MM, MAX_DISP_MM)

    # Axis mapping — matches touch_to_world
    delta = np.array([
         displacement[2],   # Touch Z → World X  (toward camera)
         displacement[0],   # Touch X → World Y  (right on screen)
         displacement[1],   # Touch Y → World Z  (up)
    ]) * RELATIVE_SPEED

    return np.clip(
        eef_pos + delta,
        ROBOT_CENTER - ROBOT_HALF_RANGE,
        ROBOT_CENTER + ROBOT_HALF_RANGE
    )


# ─────────────────────────────────────────────────────────────
# FORCE EXTRACTION
# ─────────────────────────────────────────────────────────────

_filtered_force = np.zeros(3)

# OBJECT_BODY_NAME — auto-detected at startup, set manually to None initially:
OBJECT_BODY_NAME = None   # None = auto-detect at startup

def _remap_to_touch(world_force):
    fx, fy, fz = world_force
    return np.array([fy, fz, fx])
    # Touch X = World Y 
    # Touch Y = World Z 
    # Touch Z = World X 
# def _remap_to_touch(world_force: np.ndarray) -> np.ndarray:
#     """Remap MuJoCo world frame → Touch device frame (inverse of position mapping)."""
#     fx, fy, fz = world_force
#     return np.array([
#          fx,   # MuJoCo X → Touch X
#          fz,   # MuJoCo Z → Touch Y
#         -fy,   # MuJoCo Y → Touch -Z
#     ])


def get_contact_force_mj(env) -> np.ndarray:
    """
    mj_contact mode: reads mj_contactForce on object geom contacts.
    """
    global _filtered_force
    sim = env.sim
    net_force = np.zeros(3)
    force_buf = np.zeros(6)

    try:
        body_id = sim.model.body_name2id(OBJECT_BODY_NAME)
        obj_geom_ids = {
            i for i in range(sim.model.ngeom)
            if sim.model.geom_bodyid[i] == body_id
        }
    except Exception:
        obj_geom_ids = set()

    if not obj_geom_ids or sim.data.ncon == 0:
        _filtered_force *= (1 - FORCE_ALPHA)
        return _filtered_force.copy()

    # Get table geom IDs to exclude gravity/resting contacts
    table_geom_ids = set()
    try:
        table_body_id = sim.model.body_name2id("table")
        table_geom_ids = {
            i for i in range(sim.model.ngeom)
            if sim.model.geom_bodyid[i] == table_body_id
        }
    except Exception:
        pass

    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 not in obj_geom_ids and g2 not in obj_geom_ids:
            continue

        # Skip table contacts — just gravity, not interesting for haptics
        other_geom = g2 if g1 in obj_geom_ids else g1
        if other_geom in table_geom_ids:
            continue

        mujoco.mj_contactForce(sim.model._model, sim.data._data, i, force_buf)
        contact_frame = c.frame.reshape(3, 3)
        f_world = contact_frame.T @ force_buf[:3]

        # mj_contactForce gives force ON geom1
        # if object is geom2, flip to get force ON object
        if g2 in obj_geom_ids:
            f_world = -f_world

        net_force += f_world

    touch_force = _remap_to_touch(net_force)
    _filtered_force = FORCE_ALPHA * touch_force + (1 - FORCE_ALPHA) * _filtered_force
    return _filtered_force.copy()


def get_contact_force_penetration(env) -> np.ndarray:
    """
    Penetration mode: spring-damper from contact.dist on object contacts.
    Uses EEF velocity from sim (m/s, world frame) for damping, more accurate
    than stylus velocity which is in a different space and lags behind sim motion.
    """
    global _filtered_force
    sim = env.sim
    net_force = np.zeros(3)

    # Use EEF velocity from sim (m/s, world frame) — more accurate than stylus velocity
    eef_vel = sim.data.body_xvelp[
        sim.model.body_name2id("gripper0_right_hand")
    ]

    try:
        body_id = sim.model.body_name2id(OBJECT_BODY_NAME)
        cube_geom_ids = {
            i for i in range(sim.model.ngeom)
            if sim.model.geom_bodyid[i] == body_id
        }
    except Exception:
        cube_geom_ids = set()

    if not cube_geom_ids or sim.data.ncon == 0:
        _filtered_force *= (1 - FORCE_ALPHA)
        return _filtered_force.copy()

    # Exclude table contacts — resting gravity force, not haptic feedback
    table_geom_ids = set()
    try:
        table_body_id = sim.model.body_name2id("table")
        table_geom_ids = {
            i for i in range(sim.model.ngeom)
            if sim.model.geom_bodyid[i] == table_body_id
        }
    except Exception:
        pass

    has_contact = False

    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = c.geom1, c.geom2

        if g1 not in cube_geom_ids and g2 not in cube_geom_ids:
            continue

        other_geom = g2 if g1 in cube_geom_ids else g1
        if other_geom in table_geom_ids:
            continue

        depth = -c.dist
        if depth <= 1e-4:
            continue

        has_contact = True

        # Normal points from geom1 → geom2
        # If cube is geom1, normal points INTO cube → flip to get pushback direction
        normal = c.frame[:3].copy()
        if g1 in cube_geom_ids:
            normal = -normal

        f_stiffness = STIFFNESS * depth * normal
        vel_along_normal = np.dot(eef_vel, normal)   # EEF vel, already in m/s
        f_damping = -DAMPING * vel_along_normal * normal
        f_contact = np.clip(f_stiffness + f_damping, -MAX_FORCE_N, MAX_FORCE_N)
        net_force += f_contact

    if not has_contact:
        _filtered_force *= (1 - FORCE_ALPHA)
        return _filtered_force.copy()

    touch_force = _remap_to_touch(net_force)
    _filtered_force = FORCE_ALPHA * touch_force + (1 - FORCE_ALPHA) * _filtered_force
    return _filtered_force.copy()


def get_contact_force_gripper_mj(env) -> np.ndarray:
    """
    Render forces on gripper finger geoms directly.
    Sums mj_contactForce on all contacts involving finger geoms.
    Negates result (Newton's 3rd law: we want force ON gripper, not FROM gripper).
    """
    global _filtered_force
    sim = env.sim
    net_force = np.zeros(3)
    force_buf = np.zeros(6)

    finger_ids = set()
    for name in FINGER_GEOMS:
        try:
            finger_ids.add(sim.model.geom_name2id(name))
        except Exception:
            pass

    if not finger_ids or sim.data.ncon == 0:
        _filtered_force *= (1 - FORCE_ALPHA)
        return _filtered_force.copy()

    has_contact = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 not in finger_ids and g2 not in finger_ids:
            continue
        if g1 in finger_ids and g2 in finger_ids:
            continue
        has_contact = True
        mujoco.mj_contactForce(sim.model._model, sim.data._data, i, force_buf)
        contact_frame = c.frame.reshape(3, 3)
        f_world = contact_frame.T @ force_buf[:3]
        if g2 in finger_ids:
            f_world = -f_world
        net_force += f_world

    if not has_contact:
        _filtered_force *= (1 - FORCE_ALPHA)
        return _filtered_force.copy()

    touch_force = -_remap_to_touch(net_force)  # negate: Newton's 3rd law
    _filtered_force = FORCE_ALPHA * touch_force + (1 - FORCE_ALPHA) * _filtered_force
    return _filtered_force.copy()


def get_contact_force(env, stylus_vel_mm_s: np.ndarray, gripper_closed: bool = False) -> np.ndarray:
    """
    Auto-switch force body based on gripper state:
      - Gripper open   : feel gripper finger forces directly
      - Gripper closed : feel object forces (cylinder hitting table/hole)
    """
    if FORCE_BODY == "gripper":
        return get_contact_force_gripper_mj(env)
    elif FORCE_BODY == "object":
        if FORCE_MODE == "penetration":
            return get_contact_force_penetration(env)
        else:
            return get_contact_force_mj(env)
    else:  # "auto" — switch based on gripper state
        if gripper_closed:
            # Holding object (feel what object feels)
            if FORCE_MODE == "penetration":
                return get_contact_force_penetration(env)
            else:
                return get_contact_force_mj(env)
        else:
            # Free moving (feel what gripper feels)
            return get_contact_force_gripper_mj(env)



# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from round_nut_only_env import RoundNutOnlyEnv

    print("=" * 55)
    print(f"  Robosuite Haptic Bridge  |  mode: {MAPPING_MODE}")
    print(f"  Force mode: {FORCE_MODE} | Force body: {FORCE_BODY} | Object: {OBJECT_BODY_NAME}")
    print("=" * 55)

    print("\n[1/2] Initializing robosuite...")

    ctrl_config_path = "osc_world_frame.json"
    try:
        with open(ctrl_config_path) as f:
            ctrl_config = json.load(f)
        print("  Using world frame OSC_POSE controller")
    except Exception:
        ctrl_config = load_composite_controller_config(controller="BASIC")
        print("  Using BASIC controller")

    env = RoundNutOnlyEnv(
        robots="Panda",
        controller_configs=ctrl_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agentview",
        control_freq=SIM_HZ,
        horizon=100000,
    )
    env.robots[0].init_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    obs = env.reset()

    # ── Auto-detect object body name ──
    SKIP_KEYWORDS = ['robot', 'gripper', 'table', 'floor', 'world', 'peg',
                     'visual', 'collision', 'left_eef', 'right_eef', 'square',
                     'mount', 'fixed']
    if OBJECT_BODY_NAME is None:
        for jnt in range(env.sim.model.njnt):
            if env.sim.model.jnt_type[jnt] == 0:  # freejoint
                bid = env.sim.model.jnt_bodyid[jnt]
                bname = env.sim.model.body_id2name(bid)
                if not any(k in bname.lower() for k in SKIP_KEYWORDS):
                    OBJECT_BODY_NAME = bname
                    print(f"  Auto-detected object body: {OBJECT_BODY_NAME}")
                    break
        if OBJECT_BODY_NAME is None:
            print("  WARNING: could not auto-detect object body — forces disabled")
            OBJECT_BODY_NAME = ""
    for _ in range(50):
        action = np.zeros(env.action_dim)
        action[-1] = 1.0
        obs, _, _, _ = env.step(action)

    eef_start = obs['robot0_eef_pos'].copy()
    print(f"  EEF start    : {eef_start.round(3)}")
    print("  Environment ready.")

    # Update robot center to match actual reset position
    ROBOT_CENTER[:] = eef_start

    # ── Init Touch device AFTER arm is positioned ──
    print("\n[2/2] Initializing Touch device...")
    device = HapticDevice(callback=haptic_callback, scheduler_type="async")
    hd.enable_force()
    time.sleep(0.5)  # let haptic callback populate position

    # Record stylus home position — relative mode moves from HERE not from TOUCH_CENTER
    with lock:
        stylus_home = shared.stylus_pos_mm.copy()
    print(f"  Stylus home  : {stylus_home.round(1)} mm")
    print("  Touch device ready.")

    print(f"\nControls:")
    print(f"  Move stylus  → robot end effector  [{MAPPING_MODE} mode]")
    print(f"  Button 1     → toggle gripper")
    print(f"  A           → toggle force feedback")
    print(f"  SPACEBAR     → reset simulation")
    print(f"  Ctrl+C       → quit\n")

    # ── Start keyboard listener ──
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    sim_step = 0
    with lock:
        shared.gripper_closed = False

    try:
        while shared.running:
            # ── Handle reset request ──
            if shared.reset_requested:
                with lock:
                    shared.reset_requested = False
                print("\n  Resetting simulation...")
                obs = env.reset()
                env.robots[0].init_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
                obs = env.reset()
                for _ in range(50):
                    action = np.zeros(env.action_dim)
                    action[-1] = 1.0
                    obs, _, _, _ = env.step(action)
                eef_start = obs['robot0_eef_pos'].copy()
                ROBOT_CENTER[:] = eef_start
                with lock:
                    stylus_home = shared.stylus_pos_mm.copy()
                    shared.gripper_closed = False
                print(f"  Simulation reset. EEF at {eef_start.round(3)} m\n")
                continue

            t0 = time.perf_counter()

            # ── Read shared state ──
            with lock:
                stylus_pos     = shared.stylus_pos_mm.copy()
                button_now     = shared.button1_pressed
                gripper_closed = shared.gripper_closed
                last_button    = shared.last_button1

            # ── Gripper toggle on button rising edge ──
            if button_now and not last_button:
                gripper_closed = not gripper_closed
                print(f"  Gripper {'CLOSED' if gripper_closed else 'OPEN'}")

            # ── Compute target eef position ──
            eef_pos = obs['robot0_eef_pos']

            if MAPPING_MODE == "absolute":
                target = compute_target_absolute(stylus_pos)
            else:
                target = compute_target_relative(stylus_pos, eef_pos, stylus_home)

            # ── Build action vector [dx dy dz drx dry drz gripper] ──
            pos_error_world   = target - eef_pos
            pos_error_clipped = np.clip(pos_error_world * ACTION_GAIN, -1.0, 1.0)

            action      = np.zeros(env.action_dim)
            action[:3]  = pos_error_clipped
            action[-1]  = -1.0 if gripper_closed else 1.0

            # ── Step simulation ──
            obs, _, done, _ = env.step(action)
            env.render()

            # ── Extract and send contact forces ──
            try:
                with lock:
                    force_enabled = shared.force_feedback_enabled
                    stylus_vel = shared.stylus_vel_mm_s.copy()
                if force_enabled:
                    force = get_contact_force(env, stylus_vel, gripper_closed)
                    if FORCE_MODE == "mj_contact":
                        scaled_force = np.clip(force * FORCE_SCALE, -MAX_FORCE_N, MAX_FORCE_N)
                    else:
                        scaled_force = np.clip(force, -MAX_FORCE_N, MAX_FORCE_N)
                    raw_force_mag = np.linalg.norm(force)
                    # Estimate total force sent to device — damping only when in contact
                    in_contact_estimate  = np.linalg.norm(scaled_force) > 1e-4
                    damping_estimate     = -FORCE_DAMP * stylus_vel / 1000.0 if in_contact_estimate else np.zeros(3)
                    total_force_estimate = np.clip(scaled_force + damping_estimate, -MAX_FORCE_N, MAX_FORCE_N)
                else:
                    scaled_force         = np.zeros(3)
                    raw_force_mag        = 0.0
                    damping_estimate     = np.zeros(3)
                    total_force_estimate = np.zeros(3)
            except Exception:
                scaled_force         = np.zeros(3)
                raw_force_mag        = 0.0
                damping_estimate     = np.zeros(3)
                total_force_estimate = np.zeros(3)

            with lock:
                shared.contact_force_N[:] = scaled_force
                shared.last_button1       = button_now
                shared.gripper_closed     = gripper_closed

            # ── Debug every 2s ──
            sim_step += 1
            if sim_step % (SIM_HZ * 2) == 0:
                axes = ['X', 'Y', 'Z']
                dom  = axes[np.argmax(np.abs(action[:3]))]
                print(f"  stylus XYZ={stylus_pos.round(1)} mm")
                print(f"  eef    XYZ={eef_pos.round(3)} m")
                print(f"  target XYZ={target.round(3)} m")
                print(f"  action XYZ={action[:3].round(3)}  dominant={dom}")
                print(f"  ncon={env.sim.data.ncon} | raw={raw_force_mag:.3f} | scale={FORCE_SCALE} | scaled={scaled_force.round(3)} | damping={damping_estimate.round(3)} | total≈{total_force_estimate.round(3)} N | MAX={MAX_FORCE_N}")

                # Diagnostic: print all body names with non-zero cfrc_ext
                for bid in range(env.sim.model.nbody):
                    f = env.sim.data.cfrc_ext[bid, :3]
                    if np.linalg.norm(f) > 0.1:
                        name = env.sim.model.body_id2name(bid)
                        print(f"    cfrc_ext[{name}] = {f.round(3)}")

                # Diagnostic: print cube contact forces directly
                try:
                    body_id = env.sim.model.body_name2id(OBJECT_BODY_NAME)
                    cube_geom_ids = {
                        i for i in range(env.sim.model.ngeom)
                        if env.sim.model.geom_bodyid[i] == body_id
                    }
                    force_buf = np.zeros(6)
                    for i in range(env.sim.data.ncon):
                        c = env.sim.data.contact[i]
                        g1, g2 = c.geom1, c.geom2
                        if g1 not in cube_geom_ids and g2 not in cube_geom_ids:
                            continue
                        mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, i, force_buf)
                        contact_frame = c.frame.reshape(3, 3)
                        f_world = contact_frame.T @ force_buf[:3]
                        if g2 in cube_geom_ids:
                            f_world = -f_world
                        g1n = env.sim.model.geom_id2name(g1)
                        g2n = env.sim.model.geom_id2name(g2)
                        print(f"    cube_contact: {g1n}↔{g2n} f={f_world.round(3)} mag={np.linalg.norm(f_world):.3f}")
                except Exception as e:
                    print(f"    diagnostic error: {e}")

            # Real-time contact print when gripper touches peg
            if env.sim.data.ncon > 0:
                has_peg_contact = any(
                    'cylinder_peg' in (env.sim.model.geom_id2name(env.sim.data.contact[i].geom1) or '') or
                    'cylinder_peg' in (env.sim.model.geom_id2name(env.sim.data.contact[i].geom2) or '')
                    for i in range(env.sim.data.ncon)
                    if any(g in {env.sim.model.geom_id2name(env.sim.data.contact[i].geom1),
                                 env.sim.model.geom_id2name(env.sim.data.contact[i].geom2)}
                           for g in FINGER_GEOMS)
                )
                if has_peg_contact:
                    print(f"  PEG CONTACT raw={raw_force_mag:.2f}N "
                          f"scaled={np.linalg.norm(scaled_force):.3f}N", end="\r")

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 1.0 / SIM_HZ - elapsed))

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        listener.stop()
        shared.running = False
        with lock:
            shared.contact_force_N[:] = np.zeros(3)
        time.sleep(0.2)
        device.close()
        env.close()
        print("Done.")