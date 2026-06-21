# haptic_rendering_tests
"""
Haptic Rendering Quality Analysis
=====================================
Verifies the force rendering quality of the teleoperation system
(osc_tests.py handles the robot kinematics and pose control analysis).

All three experiments were performed using the same env, controller config, contact params, and action generation logic as the teleoperation system.
These experiments all use the table as the contact surface to analyze the following:
1. Force Transparency:
2. Z-Width: max stable rendered impedance by sweeping table stiffnes via solref[0]
3. Passivity/Energy: confirms that energy grows at unstable solrefs defined by results of Exp2 above.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import robosuite_haptic_bridge as bridge
from robosuite_haptic_bridge import (
    get_contact_force_gripper_mj,
    FINGER_GEOMS,
    FORCE_SCALE,
    MAX_FORCE_N,
    ACTION_GAIN,
    SIM_HZ,
)
from round_nut_only_env import RoundNutOnlyEnv


# ─────────────────────────────────────────────────────────────
# CONFIG  (mirror haptic_bridge.py values exactly)
# ─────────────────────────────────────────────────────────────

FINGER_GEOMS = {
    'gripper0_right_finger1_collision',
    'gripper0_right_finger2_collision',
    'gripper0_right_finger1_pad_collision',
    'gripper0_right_finger2_pad_collision',
    'gripper0_right_hand_collision',
}

ACTION_GAIN  = 5.0
SIM_HZ       = 10
FORCE_ALPHA  = 1.0   # 1.0 = no LPF — matches bridge default
FORCE_SCALE  = 0.01  # same as haptic_bridge.py

OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haptic_test_results")
PLOT_DIR     = os.path.join(OUTPUT_DIR, "plots")


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def _ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def apply_solref_patches(env, table_timeconst=0.02, dampratio=2.0,
                         finger_timeconst=0.01):
    for i in range(env.sim.model.ngeom):
        name = env.sim.model.geom_id2name(i)
        if 'table' in name.lower():
            env.sim.model.geom_solref[i] = [table_timeconst, dampratio]
        elif any(g in name for g in ['finger', 'hand_collision']):
            env.sim.model.geom_solref[i] = [finger_timeconst, dampratio]


def make_env(ctrl_config, table_timeconst=0.02, sim_hz=SIM_HZ):
    env = RoundNutOnlyEnv(
        robots="Panda",
        controller_configs=ctrl_config,   # NOTE: plural, matches RoundNutOnlyEnv signature
        has_renderer=True,
        control_freq=sim_hz,
        horizon=500000,
    )
    env.robots[0].init_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    obs = env.reset()
    apply_solref_patches(env, table_timeconst=table_timeconst)
    # Settle gripper open
    for _ in range(50):
        action = np.zeros(env.action_dim)
        action[-1] = 1.0
        obs, _, _, _ = env.step(action)
        env.render()
    return env, obs


def get_finger_ids(env):
    """Resolve FINGER_GEOMS names to geom IDs. Uses FINGER_GEOMS imported from bridge."""
    ids = set()
    for name in FINGER_GEOMS:
        try:
            ids.add(env.sim.model.geom_name2id(name))
        except Exception:
            print(f"    [WARN] geom not found: {name}")
    return ids


def get_table_geom_ids(env):
    ids = set()
    try:
        table_body_id = env.sim.model.body_name2id("table")
        for i in range(env.sim.model.ngeom):
            if env.sim.model.geom_bodyid[i] == table_body_id:
                ids.add(i)
    except Exception:
        pass
    return ids


def diagnose_contacts(env, label=""):
    """Print all active contacts and their geom names — for debugging zero-force issues."""
    sim = env.sim
    print(f"  [DIAG {label}] ncon={sim.data.ncon}")
    for i in range(min(sim.data.ncon, 10)):
        c  = sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(c.geom1) or f"id{c.geom1}"
        g2 = env.sim.model.geom_id2name(c.geom2) or f"id{c.geom2}"
        print(f"    contact[{i}]: {g1} ↔ {g2}  dist={c.dist:.5f}")


def find_table_contact(env, obs, max_steps=3000, step_size=0.0002):
    """
    Drive EEF downward until a finger geom contacts a table geom.
    Uses geom-filtered contact detection — avoids false positives from
    robot self-contacts that exist at startup (ncon > 0 at t=0).

    Returns (table_contact_z, obs) or (None, obs) if not found.
    """
    finger_ids = get_finger_ids(env)
    table_ids  = get_table_geom_ids(env)

    if not finger_ids:
        print("    [ERROR] No finger geom IDs resolved — check FINGER_GEOMS names")
        return None, obs
    if not table_ids:
        print("    [ERROR] No table geom IDs found")
        return None, obs

    target = obs['robot0_eef_pos'].copy()

    for step in range(max_steps):
        target[2] -= step_size
        err    = target - obs['robot0_eef_pos']
        action = np.zeros(env.action_dim)
        action[:3] = np.clip(err * ACTION_GAIN, -1.0, 1.0)
        action[-1] = 1.0   # gripper open
        obs, _, _, _ = env.step(action)

        # Only count contact if a finger geom touches a table geom
        for ci in range(env.sim.data.ncon):
            c  = env.sim.data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            finger_in = (g1 in finger_ids) or (g2 in finger_ids)
            table_in  = (g1 in table_ids)  or (g2 in table_ids)
            if finger_in and table_in:
                z = obs['robot0_eef_pos'][2]
                print(f"    Finger-table contact at EEF z={z:.4f}m  (step {step})")
                return z, obs

    print(f"    [WARN] No finger-table contact found in {max_steps} steps")
    return None, obs


def get_rendered_force(env):
    """
    Call bridge's get_contact_force_gripper_mj() directly — exact same pipeline
    as the Touch device receives during teleoperation, including LPF state.
    Returns scaled force (N) ready for Touch, same as shared.contact_force_N in bridge.
    Call reset_bridge_force_state() between trials to clear LPF memory.
    """
    raw = get_contact_force_gripper_mj(env)
    return np.clip(raw * FORCE_SCALE, -MAX_FORCE_N, MAX_FORCE_N)


def reset_bridge_force_state():
    """Zero the bridge's LPF state between trials so each starts clean."""
    bridge._filtered_force[:] = 0.0


def get_ground_truth_force(env):
    """cfrc_ext summed over gripper/hand bodies — MuJoCo constraint force ground truth."""
    sim = env.sim
    net = np.zeros(3)
    for bid in range(sim.model.nbody):
        name = sim.model.body_id2name(bid)
        if 'gripper' in name.lower() or 'hand' in name.lower():
            net += sim.data.cfrc_ext[bid, :3]
    return net


def detect_oscillation(force_arr, threshold_cv=0.3, threshold_crossings=20):
    """CV = std/mean; high CV + many zero-crossings = oscillating signal."""
    steady = force_arr[int(0.5 * len(force_arr)):]
    if len(steady) < 10 or steady.mean() < 1e-6:
        return False, 0, 0.0
    centered  = steady - steady.mean()
    crossings = int(np.sum(np.diff(np.sign(centered)) != 0))
    cv        = steady.std() / (steady.mean() + 1e-9)
    return (crossings > threshold_crossings) or (cv > threshold_cv), crossings, float(cv)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 1 — Force Transparency
# ─────────────────────────────────────────────────────────────

def run_force_transparency_test(ctrl_config, n_trials=5,
                                 hold_overshoot_m=0.010, duration_contact=3.0,
                                 table_timeconst=0.02):
    """
    n_trials identical trials: command EEF to fixed 10mm below table surface and hold.
    Logs rendered force (what Touch device receives) vs. MuJoCo ground-truth contact force.
    Normalized overlay: perfect transparency = profiles overlap across all trials.
    Bottom panel: absolute error between rendered and ground-truth force over time.
    """
    _ensure_dirs()
    print(f"\n[Exp1 Transparency] {n_trials} trials, "
          f"overshoot={hold_overshoot_m*1000:.0f}mm, table_timeconst={table_timeconst}")
    all_rows = []

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials} ...", end=" ", flush=True)
        env, obs      = make_env(ctrl_config, table_timeconst=table_timeconst)
        table_z, obs  = find_table_contact(env, obs)

        if table_z is None:
            diagnose_contacts(env, label=f"trial{trial}_no_contact")
            env.close()
            continue

        reset_bridge_force_state()   # clear LPF between trials
        target        = obs['robot0_eef_pos'].copy()
        target[2]     = table_z - hold_overshoot_m   # fixed 10mm below table — same every trial
        contact_steps = int(duration_contact * SIM_HZ)

        for k in range(contact_steps):
            t       = k / SIM_HZ
            eef_pos = obs['robot0_eef_pos']

            err    = target - eef_pos
            action = np.zeros(env.action_dim)
            action[:3] = np.clip(err * ACTION_GAIN, -1.0, 1.0)
            action[-1] = 1.0
            obs, _, _, _ = env.step(action)
            # env.render()

            f_rendered = get_rendered_force(env)
            f_gt       = get_ground_truth_force(env)

            all_rows.append({
                "trial":          trial,
                "t":              t,
                "f_rendered_mag": np.linalg.norm(f_rendered),
                "f_gt_mag":       np.linalg.norm(f_gt),
                "f_error":        np.linalg.norm(f_rendered - f_gt),
                "ncon":           env.sim.data.ncon,
            })

        print("done")
        env.close()

    if not all_rows:
        print("  [ERROR] No data collected — check contact detection")
        return None

    df = pd.DataFrame(all_rows)
    for col, norm_col in [("f_rendered_mag", "f_rendered_norm"),
                           ("f_gt_mag",       "f_gt_norm")]:
        df[norm_col] = df.groupby("trial")[col].transform(
            lambda x: x / (x.max() + 1e-9))

    path = os.path.join(OUTPUT_DIR, "force_transparency.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}  ({len(df)} rows, {df['trial'].nunique()} trials)")
    return df


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 2 — Z-Width via solref Sweep
# ─────────────────────────────────────────────────────────────

def run_zwidth_solref_sweep(ctrl_config,
                             timeconst_values=None,
                             overshoot_depths_m=None,
                             csv_name="zwidth_solref_sweep"):
    """
    Sweep table solref[0] (timeconst) to characterize rendered impedance and stability.

    Key insight: MuJoCo's contact constraints prevent real geometric penetration,
    so c.dist stays ~0 regardless of solref. Instead we use COMMANDED overshoot —
    how far below the table surface we command the EEF — as the input variable.
    The rendered force at a given commanded overshoot tells us the effective
    rendered stiffness: K_eff = F_steady / overshoot_depth.

    Instability detection uses the TRANSIENT (first 0.5s after contact):
      - Peak overshoot ratio: F_peak / F_steady  (>1.5 = ringing)
      - Transient zero-crossings of (F - F_steady): many = oscillating

    Colgate-Schenkel bound (sampled-data haptics):
      K_eff < 2 * B_virtual / T,  T = 1/SIM_HZ
    We report K_eff vs. the theoretical bound for comparison.
    """
    _ensure_dirs()
    if timeconst_values is None:
        timeconst_values = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    if overshoot_depths_m is None:
        overshoot_depths_m = [0.005]   # 5mm commanded below table — fixed for sweep

    T   = 1.0 / SIM_HZ
    print(f"\n[Exp2 Z-Width solref Sweep]  T={T*1000:.1f}ms")

    all_rows   = []
    force_logs = {}   # (tc, depth) → force array

    for tc in timeconst_values:
        for overshoot in overshoot_depths_m:
            key = (tc, overshoot)
            print(f"  solref[0]={tc:.4f}  overshoot={overshoot*1000:.1f}mm ... ",
                  end="", flush=True)

            env, obs     = make_env(ctrl_config, table_timeconst=tc)
            table_z, obs = find_table_contact(env, obs)

            if table_z is None:
                diagnose_contacts(env, label=f"tc{tc}_no_contact")
                env.close()
                print("no contact — skipping")
                continue

            # Command EEF to fixed overshoot below table surface
            reset_bridge_force_state()   # clear LPF for each condition
            target         = obs['robot0_eef_pos'].copy()
            target[2]      = table_z - overshoot
            total_steps    = int(2.5 * SIM_HZ)
            transient_end  = int(0.5 * SIM_HZ)   # first 0.5s = transient
            force_arr      = []

            for k in range(total_steps):
                eef_pos    = obs['robot0_eef_pos']
                err        = target - eef_pos
                action     = np.zeros(env.action_dim)
                action[:3] = np.clip(err * ACTION_GAIN, -1.0, 1.0)
                action[-1] = 1.0
                obs, _, _, _ = env.step(action)
                # env.render()

                f_rendered = get_rendered_force(env)
                force_arr.append(np.linalg.norm(f_rendered))

            force_arr  = np.array(force_arr)
            force_logs[key] = force_arr

            # Steady-state: last 1s
            steady_arr = force_arr[int(1.5 * SIM_HZ):]
            F_steady   = steady_arr.mean() if len(steady_arr) > 0 else 1e-9

            # K_eff: rendered force per unit commanded overshoot
            K_eff = F_steady / overshoot if overshoot > 0 else 0.0

            # Transient oscillation: first 0.5s after contact
            transient_arr = force_arr[:transient_end]
            F_peak        = transient_arr.max() if len(transient_arr) > 0 else 0.0
            overshoot_ratio = F_peak / (F_steady + 1e-9)

            centered   = transient_arr - F_steady
            crossings  = int(np.sum(np.diff(np.sign(centered)) != 0))

            # Instability: large overshoot ratio OR sustained oscillation in transient
            oscillating = (overshoot_ratio > 1.5) or (crossings > 10)

            # Colgate-Schenkel theoretical bound (assumes B_virtual ≈ solref dampratio × K_eff × T)
            # Simplified: report K_eff vs K_max = 2*B/T where B is estimated from settling
            # Use dampratio=2.0 (our solref[1]) as a proxy: B_proxy = dampratio * T * K_eff
            B_proxy = 2.0 * T * K_eff
            K_max_theory = 2 * B_proxy / T   # = 4 * K_eff (always satisfied — informational)

            status = "UNSTABLE" if oscillating else "stable "
            print(f"{status}  K_eff={K_eff:.2f} N/m  F_steady={F_steady:.5f}N  "
                  f"overshoot_ratio={overshoot_ratio:.2f}  crossings={crossings}")

            all_rows.append({
                "table_timeconst":  tc,
                "overshoot_m":      overshoot,
                "K_eff_Nm":         K_eff,
                "F_steady_N":       float(F_steady),
                "F_peak_N":         float(F_peak),
                "overshoot_ratio":  float(overshoot_ratio),
                "transient_crossings": crossings,
                "oscillating":      oscillating,
                "K_max_theory_Nm":  K_max_theory,
            })
            env.close()

    df = pd.DataFrame(all_rows)
    path = os.path.join(OUTPUT_DIR, f"{csv_name}.csv")
    df.to_csv(path, index=False)

    # Save traces — one column per (tc, overshoot) key
    max_len  = max(len(v) for v in force_logs.values()) if force_logs else 0
    trace_df = pd.DataFrame({
        f"tc_{tc}_ov_{ov}": np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        for (tc, ov), arr in force_logs.items()
    })
    trace_df.to_csv(os.path.join(OUTPUT_DIR, f"{csv_name}_traces.csv"), index=False)
    print(f"\n  Saved {path}")
    return df, force_logs


# ─────────────────────────────────────────────────────────────
# EXPERIMENT 3 — Passivity / Energy Monitor
# ─────────────────────────────────────────────────────────────

def run_passivity_test(ctrl_config, table_timeconst=0.02,
                        duration=5.0, label=""):
    """
    Hold EEF at 3mm penetration. Monitor ∫ F_rendered · v_eef dt.
    Passive system: cumulative energy stays bounded.
    Active system: energy grows → instability.
    No Touch device needed — purely sim-side computation.

    Intended usage: call once with stable tc (from exp2), once with unstable tc.
    """
    _ensure_dirs()
    tag = label or f"tc{table_timeconst}"
    print(f"\n[Exp3 Passivity] solref[0]={table_timeconst}  label={tag}")

    env, obs      = make_env(ctrl_config, table_timeconst=table_timeconst)
    table_z, obs  = find_table_contact(env, obs)

    if table_z is None:
        diagnose_contacts(env, label=f"{tag}_no_contact")
        env.close()
        return None

    reset_bridge_force_state()   # clean LPF state
    target    = obs['robot0_eef_pos'].copy()
    target[2] = table_z - 0.003
    dt        = 1.0 / SIM_HZ

    cumulative_energy = 0.0
    log = []

    for k in range(int(duration * SIM_HZ)):
        t       = k * dt
        eef_pos = obs['robot0_eef_pos']
        eef_vel = env.sim.data.get_body_xvelp("robot0_right_hand")

        f_scaled = get_rendered_force(env)   # exact value Touch device would receive

        power              = np.dot(f_scaled, eef_vel)
        cumulative_energy += power * dt

        err    = target - eef_pos
        action = np.zeros(env.action_dim)
        action[:3] = np.clip(err * ACTION_GAIN, -1.0, 1.0)
        action[-1] = 1.0
        obs, _, _, _ = env.step(action)

        log.append({
            "t":                   t,
            "label":               tag,
            "table_timeconst":     table_timeconst,
            "eef_vel_z":           eef_vel[2],
            "f_rendered_mag":      np.linalg.norm(f_scaled),
            "power_W":             power,
            "cumulative_energy_J": cumulative_energy,
            "ncon":                env.sim.data.ncon,
        })

    env.close()
    df = pd.DataFrame(log)
    path = os.path.join(OUTPUT_DIR, f"passivity_{tag}.csv")
    df.to_csv(path, index=False)

    final_e = df["cumulative_energy_J"].iloc[-1]
    verdict = "ACTIVE (growing)" if final_e > 0.01 else "passive (bounded)"
    print(f"  Cumulative energy: {final_e:.6f} J  →  {verdict}")
    print(f"  Peak power:        {df['power_W'].abs().max():.6f} W")
    print(f"  Saved {path}")
    return df


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_transparency(csv_path=None):
    """
    Panel 1: Normalized force profiles (rendered solid, GT dashed) — all trials overlaid.
    Panel 2: Raw force error vs. penetration depth.
    """
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, "force_transparency.csv")
    df     = pd.read_csv(csv_path)
    trials = df["trial"].unique()
    colors = cm.tab10(np.linspace(0, 1, len(trials)))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    fig.suptitle("Exp 1 — Force Transparency\n"
                 "Normalized rendered force vs. ground-truth (cfrc_ext)", fontsize=12)

    ax = axes[0]
    for i, trial in enumerate(trials):
        td = df[df["trial"] == trial]
        ax.plot(td["t"], td["f_rendered_norm"], color=colors[i], lw=1.5)
        ax.plot(td["t"], td["f_gt_norm"],       color=colors[i], lw=1.5, ls="--", alpha=0.6)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="k", lw=2,           label="rendered (solid)"),
        Line2D([0], [0], color="k", lw=2, ls="--",  label="ground truth (dashed)"),
    ], fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Force")
    ax.set_title("Normalized Force Profile per Trial — Overlap = High Transparency")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i, trial in enumerate(trials):
        td = df[df["trial"] == trial]
        ax2.plot(td["t"], td["f_error"],
                 color=colors[i], lw=1.2, alpha=0.8, label=f"trial {trial}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Force Error Magnitude (N)\n"
                   "(how far rendered force deviates from\n"
                   "MuJoCo ground-truth contact force)")
    ax2.set_title("Absolute Force Tracking Error Over Time\n"
                  "Lower = haptic device receives more accurate force")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "exp1_transparency.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved: {out}")


def plot_zwidth_sweep(summary_csv=None, traces_csv=None):
    """
    Panel 1: Force time traces — transient + steady state, red=unstable, blue=stable.
             Vertical dashed line marks end of transient window (0.5s).
    Panel 2: F_steady vs solref — effective impedance curve across surface settings.
    Panel 3: Overshoot ratio and transient zero-crossings — instability onset markers.
    """
    if summary_csv is None:
        summary_csv = os.path.join(OUTPUT_DIR, "zwidth_solref_sweep.csv")
    if traces_csv is None:
        traces_csv  = os.path.join(OUTPUT_DIR, "zwidth_solref_sweep_traces.csv")

    df_sum    = pd.read_csv(summary_csv)
    df_traces = pd.read_csv(traces_csv)
    t_axis    = np.arange(len(df_traces)) / SIM_HZ

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.suptitle("Exp 2 — Z-Width: Table Contact Stiffness (solref[0]) Sweep\n"
                 "Instability detected via transient overshoot ratio",
                 fontsize=12)

    # Panel 1: force time traces
    ax = axes[0]
    for col in df_traces.columns:
        parts    = col.split("_")   # tc_{tc}_ov_{ov}
        tc       = float(parts[1])
        row      = df_sum[df_sum["table_timeconst"] == tc]
        unstable = bool(row["oscillating"].values[0]) if len(row) else False
        ax.plot(t_axis, df_traces[col].values,
                color="crimson" if unstable else "steelblue",
                ls="--" if unstable else "-", lw=1.2,
                label=f"tc={tc}{'  ⚠' if unstable else ''}")
    ax.axvline(0.5, color="gray", ls=":", lw=1.2, label="transient end")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rendered Force × FORCE_SCALE  (N)")
    ax.set_title("Force Time Traces — Blue = stable, Red = unstable")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: F_steady vs solref (impedance curve)
    ax2      = axes[1]
    colors_p = ["crimson" if u else "steelblue" for u in df_sum["oscillating"]]
    ax2.bar(range(len(df_sum)), df_sum["F_steady_N"], color=colors_p,
            edgecolor="k", linewidth=0.5, label="F_steady")
    ax2.bar(range(len(df_sum)), df_sum["F_peak_N"], color=colors_p,
            edgecolor="k", linewidth=0.5, alpha=0.35, label="F_peak")
    ax2.set_xticks(range(len(df_sum)))
    ax2.set_xticklabels([f"{tc:.4f}" for tc in df_sum["table_timeconst"]],
                         rotation=30, ha="right")
    ax2.set_xlabel("table solref[0]  (s)   ← stiffer")
    ax2.set_ylabel("Force (N)  [after FORCE_SCALE]")
    ax2.set_title("Rendered Force: Steady-State vs. Peak — Dark = steady, Light = peak")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: overshoot ratio + transient crossings
    ax3  = axes[2]
    ax3b = ax3.twinx()
    x = range(len(df_sum))
    ax3.bar(x, df_sum["overshoot_ratio"], color=colors_p, alpha=0.7,
            label="overshoot ratio  F_peak/F_steady")
    ax3.axhline(1.5, color="orange", lw=1.5, ls="--", label="instability threshold (1.5)")
    ax3b.plot(x, df_sum["transient_crossings"], "ko--", lw=1.5, ms=5,
              label="transient zero-crossings")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels([f"{tc:.4f}" for tc in df_sum["table_timeconst"]],
                         rotation=30, ha="right")
    ax3.set_xlabel("table solref[0]  (s)")
    ax3.set_ylabel("Overshoot Ratio  F_peak / F_steady")
    ax3b.set_ylabel("Transient Zero-crossings")
    ax3.set_title("Transient Oscillation Metrics — Instability Onset")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "exp2_zwidth_sweep.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved: {out}")


def plot_passivity(csv_paths=None):
    """
    Panel 1: Cumulative energy over time — bounded = passive, growing = active.
    Panel 2: Instantaneous power F · v.
    """
    if csv_paths is None:
        files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith("passivity_"))
        csv_paths = [os.path.join(OUTPUT_DIR, f) for f in files]

    if not csv_paths:
        print("  No passivity CSVs found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("Exp 3 — Passivity: Cumulative Energy ∫ F·v dt\n"
                 "Bounded = passive (stable) | Growing = active (unstable)", fontsize=12)

    for path in csv_paths:
        df       = pd.read_csv(path)
        label    = df["label"].iloc[0]
        tc       = df["table_timeconst"].iloc[0]
        final_e  = df["cumulative_energy_J"].iloc[-1]
        unstable = final_e > 0.01
        color    = "crimson" if unstable else "steelblue"
        ls       = "--" if unstable else "-"
        tag      = f"tc={tc}  {'⚠ UNSTABLE' if unstable else 'stable'}"

        axes[0].plot(df["t"], df["cumulative_energy_J"],
                     color=color, ls=ls, lw=2, label=tag)
        axes[1].plot(df["t"], df["power_W"],
                     color=color, ls=ls, lw=1.2, alpha=0.8, label=tag)

    for ax in axes:
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cumulative Energy  (J)")
    axes[0].set_title("∫ F_rendered · v_eef  dt")
    axes[1].set_ylabel("Instantaneous Power  (W)")
    axes[1].set_xlabel("Time  (s)")
    axes[1].set_title("F_rendered · v_eef")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "exp3_passivity.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved: {out}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctrl_config_path = "osc_world_frame.json"
    with open(ctrl_config_path) as f:
        ctrl_config = json.load(f)
    print("  Using world frame OSC_POSE controller")

    # ── Exp 1: Force Transparency ──
    run_force_transparency_test(ctrl_config, n_trials=5, table_timeconst=0.02)
    plot_transparency()

    # ── Exp 2: Z-Width via solref sweep ──
    df_zwidth, _ = run_zwidth_solref_sweep(
        ctrl_config,
        timeconst_values=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        overshoot_depths_m=[0.005],
    )
    plot_zwidth_sweep()

    # ── Exp 3: Passivity — auto-select stable + unstable tc from exp2 ──
    stable_vals   = df_zwidth[~df_zwidth["oscillating"]]["table_timeconst"].values
    unstable_vals = df_zwidth[ df_zwidth["oscillating"]]["table_timeconst"].values

    if len(stable_vals) > 0:
        tc_s = float(stable_vals[-1])   # most aggressive stable value
        run_passivity_test(ctrl_config, table_timeconst=tc_s,
                           label=f"stable_tc{tc_s}")
    else:
        print("  [WARN] No stable values found in exp2 — running at default tc=0.02")
        run_passivity_test(ctrl_config, table_timeconst=0.02, label="stable_tc0.02")

    if len(unstable_vals) > 0:
        tc_u = float(unstable_vals[0])  # least aggressive unstable value
        run_passivity_test(ctrl_config, table_timeconst=tc_u,
                           label=f"unstable_tc{tc_u}")
    else:
        print("  [WARN] No unstable values found in exp2 — all timeconst values stable")

    plot_passivity()

    print("\nAll experiments complete.")
    print(f"  Results: {OUTPUT_DIR}/")
    print(f"  Plots:   {PLOT_DIR}/")