"""
haptic_simhz_sweep.py
=====================
Scripted table-impact experiment across simulation frequencies.

The haptic callback (1kHz) reads shared.contact_force_N which is only
updated at SIM_HZ. At low SIM_HZ the haptic device holds a stale force
for up to 1/SIM_HZ seconds — producing staircase artifacts and lag.

This script runs a scripted gripper-table impact at each SIM_HZ,
records the raw MuJoCo contact forces, then reconstructs what the
1kHz haptic device would have actually rendered (zero-order hold).

No haptic hardware required.

Outputs (saved to CWD):
  haptic_force_simhz_<hz>.csv      — per-trial sim-rate log
  figures/haptic_force_panels.png  — 4-panel time-domain comparison
  figures/haptic_force_ensemble.png— all SIM_HZ overlaid at contact onset
  figures/haptic_force_rms.png     — RMS error vs SIM_HZ bar chart
"""

import json
import os
import sys

import mujoco
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from round_nut_only_env import RoundNutOnlyEnv

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SIM_HZ_SWEEP   = [10, 20, 50, 100]
HAPTIC_HZ      = 1000        # Touch device scheduler rate (Hz)
ACTION_GAIN    = 5.0

DURATION       = 7.0         # total trial duration (s)
APPROACH_START = 1.0         # start driving down at this time (s)
APPROACH_RAMP  = 0.3         # ramp duration for Z descent (s)
APPROACH_DEPTH = 0.25        # target depth below home (m); table stops the arm
RETRACT_START  = 4.5         # retract back to home at this time (s)

FORCE_SCALE    = 0.01        # same scaling as robosuite_haptic_bridge.py
MAX_FORCE_N    = 2.0         # haptic device clip (N)

FIGURES_DIR    = "figures"

FINGER_GEOMS = {
    "gripper0_right_finger1_collision",
    "gripper0_right_finger2_collision",
    "gripper0_right_finger1_pad_collision",
    "gripper0_right_finger2_pad_collision",
    "gripper0_right_hand_collision",
}

COLORS = {10: "#d62728", 20: "#ff7f0e", 50: "#2ca02c", 100: "#1f77b4"}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def apply_solref_patches(env):
    for i in range(env.sim.model.ngeom):
        name = env.sim.model.geom_id2name(i)
        if "table" in name.lower():
            env.sim.model.geom_solref[i] = [0.02, 2.0]
        elif any(g in name for g in ["finger", "hand_collision"]):
            env.sim.model.geom_solref[i] = [0.01, 2.0]


def get_gripper_contact_force(env):
    """Net contact force on gripper geoms (world frame, N). Newton's 3rd law applied."""
    sim = env.sim
    force_buf = np.zeros(6)
    net = np.zeros(3)

    finger_ids = set()
    for name in FINGER_GEOMS:
        try:
            finger_ids.add(sim.model.geom_name2id(name))
        except Exception:
            pass

    if not finger_ids or sim.data.ncon == 0:
        return net

    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 not in finger_ids and g2 not in finger_ids:
            continue
        if g1 in finger_ids and g2 in finger_ids:
            continue
        mujoco.mj_contactForce(sim.model._model, sim.data._data, i, force_buf)
        frame = c.frame.reshape(3, 3)
        f_world = frame.T @ force_buf[:3]
        if g2 in finger_ids:
            f_world = -f_world
        net += f_world

    return -net  # force the surface exerts ON the gripper


def reconstruct_haptic_zoh(sim_times, haptic_cmd, haptic_hz=1000):
    """
    Zero-order hold: simulate the 1kHz haptic signal given sim-rate updates.
    Between sim steps, the haptic device holds the last received force value.
    Returns (t_haptic, force_haptic).
    """
    if len(sim_times) == 0:
        return np.array([]), np.array([])

    t_end    = sim_times[-1]
    t_haptic = np.arange(0, t_end + 1.0 / haptic_hz, 1.0 / haptic_hz)
    f_haptic = np.zeros(len(t_haptic))

    sim_idx  = 0
    current  = 0.0
    for j, t in enumerate(t_haptic):
        while sim_idx < len(sim_times) and sim_times[sim_idx] <= t:
            current = haptic_cmd[sim_idx]
            sim_idx += 1
        f_haptic[j] = current

    return t_haptic, f_haptic


# ─────────────────────────────────────────────────────────────
# TRIAL RUNNER
# ─────────────────────────────────────────────────────────────

def run_trial(ctrl_config, sim_hz):
    """Scripted table-impact at sim_hz. Returns logged data dict."""
    print(f"  [{sim_hz:3d} Hz] initializing ...", end=" ", flush=True)

    env = RoundNutOnlyEnv(
        robots="Panda",
        controller_configs=ctrl_config,
        has_renderer=True,
        control_freq=sim_hz,
        horizon=100000,
    )
    env.robots[0].init_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    obs = env.reset()
    apply_solref_patches(env)

    for _ in range(50):
        action = np.zeros(env.action_dim)
        action[-1] = 1.0
        obs, _, _, _ = env.step(action)

    eef_start = obs["robot0_eef_pos"].copy()
    print(f"EEF start {eef_start.round(3)}")

    dt        = 1.0 / sim_hz
    num_steps = int(DURATION / dt)

    sim_times  = []
    mj_force   = []   # raw MuJoCo contact force magnitude (N)
    haptic_cmd = []   # scaled force sent to haptic (N, clipped)
    ncon_log   = []

    for k in range(num_steps):
        t       = k * dt
        eef_pos = obs["robot0_eef_pos"].copy()

        # Trajectory: hold → ramp down → hold in contact → retract
        if t < APPROACH_START:
            target = eef_start.copy()
        elif t < APPROACH_START + APPROACH_RAMP:
            frac   = (t - APPROACH_START) / APPROACH_RAMP
            target = eef_start.copy()
            target[2] -= frac * APPROACH_DEPTH
        elif t < RETRACT_START:
            target    = eef_start.copy()
            target[2] -= APPROACH_DEPTH
        else:
            target = eef_start.copy()

        action       = np.zeros(env.action_dim)
        action[:3]   = np.clip((target - eef_pos) * ACTION_GAIN, -1.0, 1.0)
        action[-1]   = 1.0  # gripper open

        obs, _, _, _ = env.step(action)
        env.render()

        raw_force  = get_gripper_contact_force(env)
        raw_norm   = np.linalg.norm(raw_force)
        scaled     = float(np.clip(raw_norm * FORCE_SCALE, 0.0, MAX_FORCE_N))

        sim_times.append(t)
        mj_force.append(raw_norm)
        haptic_cmd.append(scaled)
        ncon_log.append(int(env.sim.data.ncon))

    env.close()

    sim_times  = np.array(sim_times)
    mj_force   = np.array(mj_force)
    haptic_cmd = np.array(haptic_cmd)

    t_hap, f_hap = reconstruct_haptic_zoh(sim_times, haptic_cmd, HAPTIC_HZ)

    return {
        "sim_hz":      sim_hz,
        "sim_times":   sim_times,
        "mj_force":    mj_force,
        "haptic_cmd":  haptic_cmd,
        "t_haptic":    t_hap,
        "haptic_zoh":  f_hap,
        "ncon":        np.array(ncon_log),
        "eef_start":   eef_start,
    }


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
T_BEFORE_CONTACT = 0.1   # s before contact onset to show
T_AFTER_CONTACT  = 0.5   # s after contact onset — zoom in on transient, not steady state


def _contact_onset(d):
    """Time when gripper force first becomes nonzero after APPROACH_START.
    Uses haptic_cmd > 0 rather than ncon > 0 because ncon fires for any scene
    contact (nut on table, peg, etc.) long before the gripper touches anything."""
    mask = (d["sim_times"] >= APPROACH_START) & (d["haptic_cmd"] > 0.0)
    idx  = np.where(mask)[0]
    return d["sim_times"][idx[0]] if len(idx) > 0 else APPROACH_START + APPROACH_RAMP


def plot_panels(all_data):
    """4-panel subplots: MuJoCo update stems + haptic ZOH line for each SIM_HZ."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()

    for ax, (sim_hz, d) in zip(axes, all_data.items()):
        color  = COLORS[sim_hz]
        t0     = _contact_onset(d)
        t_lo   = max(0, t0 - T_BEFORE_CONTACT)
        t_hi   = t0 + T_AFTER_CONTACT

        t_sim  = d["sim_times"]
        f_cmd  = d["haptic_cmd"]
        t_hap  = d["t_haptic"]
        f_zoh  = d["haptic_zoh"]

        ms  = (t_sim >= t_lo) & (t_sim <= t_hi)
        mh  = (t_hap >= t_lo) & (t_hap <= t_hi)

        # Haptic ZOH signal (what user feels)
        ax.step(t_hap[mh] - t0, f_zoh[mh],
                color=color, linewidth=1.2, alpha=0.85,
                label=f"Haptic 1kHz ZOH\n(step every {1000/sim_hz:.0f} ms)")

        # MuJoCo sim updates (vertical stems)
        for ts, fs in zip(t_sim[ms] - t0, f_cmd[ms]):
            ax.vlines(ts, 0, fs, color="k", linewidth=0.7, alpha=0.4)
        ax.plot([], [], color="k", linewidth=0.7, alpha=0.4, label="MuJoCo sim update")

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Contact onset")
        ax.set_title(f"SIM_HZ = {sim_hz} Hz   |   update interval = {1000/sim_hz:.0f} ms",
                     fontsize=10)
        ax.set_xlabel("Time relative to contact (s)")
        ax.set_ylabel("Haptic Force (N)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Haptic Force Rendering Quality vs Simulation Frequency\n"
        "The Touch device runs at 1 kHz but only gets new forces at SIM_HZ.\n"
        "Lower SIM_HZ → coarser staircase, longer stale hold between updates.",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "haptic_force_panels.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_rendered_force(all_data):
    """
    All SIM_HZ overlaid on one axis, aligned to t=0 = gripper contacts table.
    Y-axis is absolute force in N so amplitude differences are visible.
    Shows staircase coarseness at each rate side-by-side for direct comparison.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for sim_hz, d in all_data.items():
        t0    = _contact_onset(d)
        t_hap = d["t_haptic"]
        f_zoh = d["haptic_zoh"]
        mask  = (t_hap >= t0 - T_BEFORE_CONTACT) & (t_hap <= t0 + T_AFTER_CONTACT)

        print(f"    rendered force: {sim_hz} Hz  contact onset = {t0:.3f}s  "
              f"points in window = {mask.sum()}")

        ax.step(t_hap[mask] - t0, f_zoh[mask],
                color=COLORS[sim_hz],
                label=f"{sim_hz} Hz  (step every {1000/sim_hz:.0f} ms)",
                linewidth=1.4, alpha=0.85)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, label="Contact onset (t=0)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time relative to gripper contact onset (s)")
    ax.set_ylabel("Touch Device Rendered Force (N)")
    ax.set_title("Force Rendered on Touch Haptic Device vs Simulation Frequency\n"
                 "t=0 = gripper first contacts table  |  lower SIM_HZ = coarser staircase")
    ax.legend(title="Simulation rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "haptic_force_rendered.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_rms(all_data):
    """
    Absolute RMS: each trial's haptic ZOH vs linear interpolation of its own
    MuJoCo forces (smooth ground truth from the same trial).
    100Hz gets a small but nonzero score — its 10ms staircase still deviates
    from a smooth signal. Lower is better.
    """
    rms_errors = {}
    for sim_hz, d in all_data.items():
        t_sim  = d["sim_times"]
        f_mj   = d["haptic_cmd"]   # MuJoCo force at sim_hz
        t_hap  = d["t_haptic"]
        f_zoh  = d["haptic_zoh"]

        t_common = np.arange(t_hap[0], t_hap[-1], 1.0 / HAPTIC_HZ)
        # Smooth ground truth: linear interpolation of sim-rate MuJoCo samples
        f_truth  = np.interp(t_common, t_sim, f_mj)
        f_haptic = np.interp(t_common, t_hap, f_zoh)
        rms_errors[sim_hz] = float(np.sqrt(np.mean((f_haptic - f_truth) ** 2)))

    hz_vals  = sorted(rms_errors)
    rms_vals = [rms_errors[h] for h in hz_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(h) for h in hz_vals], rms_vals,
                  color=[COLORS[h] for h in hz_vals], width=0.5)
    ax.bar_label(bars, fmt="%.5f N", fontsize=8)
    ax.set_xlabel("Simulation Frequency (Hz)")
    ax.set_ylabel("RMS Staircase Error vs Smooth Ground Truth (N)")
    ax.set_title("Haptic Force Fidelity vs Simulation Rate\n"
                 "(ZOH staircase vs linear-interpolated MuJoCo — lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "haptic_force_rms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    print(f"\n  Absolute RMS staircase error:")
    for h, r in zip(hz_vals, rms_vals):
        print(f"    {h:3d} Hz : {r:.5f} N")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    with open("osc_world_frame.json") as f:
        ctrl_config = json.load(f)
    print("Controller: osc_world_frame.json\n")

    all_data = {}

    for sim_hz in SIM_HZ_SWEEP:
        print(f"=== Trial @ {sim_hz} Hz ===")
        d = run_trial(ctrl_config, sim_hz)
        all_data[sim_hz] = d

        pd.DataFrame({
            "t_sim":        d["sim_times"],
            "mj_force_N":   d["mj_force"],
            "haptic_cmd_N": d["haptic_cmd"],
            "ncon":         d["ncon"],
        }).to_csv(f"haptic_force_simhz_{sim_hz}.csv", index=False)
        t0 = _contact_onset(d)
        print(f"  contact onset = {t0:.3f}s  "
              f"peak mj_force = {d['mj_force'].max():.3f} N  "
              f"peak haptic_cmd = {d['haptic_cmd'].max():.4f} N  "
              f"nonzero haptic steps = {(d['haptic_cmd'] > 0).sum()}")
        print(f"  Saved haptic_force_simhz_{sim_hz}.csv\n")

    print("Generating plots ...")
    plot_panels(all_data)
    plot_rendered_force(all_data)
    plot_rms(all_data)

    print("\nDone.")
