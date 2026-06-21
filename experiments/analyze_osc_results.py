# analyze_osc_results.py

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def get_axis_from_filename(filename):
    if "_x" in filename:
        return "x"
    elif "_y" in filename:
        return "y"
    elif "_z" in filename:
        return "z"
    else:
        return "x"


# ============================================================
# STEP RESPONSE ANALYSIS
# ============================================================

def compute_step_metrics(df, axis="x"):

    t = df["t"].values

    target = df[f"target_pos_{axis}"].values
    actual = df[f"eef_pos_{axis}"].values

    step_start_idx = np.where(np.abs(np.diff(target)) > 1e-6)[0]

    if len(step_start_idx) == 0:
        return None

    step_idx = step_start_idx[0] + 1

    target_initial = target[step_idx - 1]
    target_final = target[-1]

    step_size = target_final - target_initial

    if abs(step_size) < 1e-6:
        return None

    response = actual - target_initial

    y10 = 0.10 * step_size
    y90 = 0.90 * step_size

    rise_time = np.nan

    try:
        t10 = t[np.where(response >= y10)[0][0]]
        t90 = t[np.where(response >= y90)[0][0]]
        rise_time = t90 - t10
    except:
        pass

    peak = np.max(actual)

    overshoot_percent = (
        (peak - target_final) / abs(step_size)
    ) * 100

    error = target - actual

    rms_error = rms(error)

    settling_time = np.nan

    band = 0.02 * abs(step_size)

    for i in range(step_idx, len(actual)):
        if np.all(np.abs(actual[i:] - target_final) < band):
            settling_time = t[i] - t[step_idx]
            break

    return {
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "overshoot_percent": overshoot_percent,
        "rms_error_m": rms_error,
    }


def plot_step_response(csv_file):

    df = pd.read_csv(csv_file)

    axis = get_axis_from_filename(csv_file)

    t = df["t"]

    target = df[f"target_pos_{axis}"]
    actual = df[f"eef_pos_{axis}"]

    plt.figure(figsize=(8,4))
    plt.plot(t, target, label="Target")
    plt.plot(t, actual, label="EEF")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(os.path.basename(csv_file))

    plt.grid(True)
    plt.legend()

    save_name = os.path.basename(csv_file).replace(".csv", ".png")

    plt.savefig(
        os.path.join(FIGURES_DIR, save_name),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def analyze_step_responses():

    rows = []

    for csv_file in sorted(glob.glob(os.path.join(DATA_DIR, "step_response_*.csv"))):

        df = pd.read_csv(csv_file)

        axis = get_axis_from_filename(csv_file)

        metrics = compute_step_metrics(df, axis)

        if metrics is None:
            continue

        metrics["file"] = csv_file

        rows.append(metrics)

        plot_step_response(csv_file)

    summary = pd.DataFrame(rows)

    summary.to_csv(
        os.path.join(DATA_DIR, "summary_stepResponse.csv"),
        index=False
    )

    print(summary)


# ============================================================
# FREQUENCY RESPONSE ANALYSIS
# ============================================================

def estimate_phase_lag(df):

    target = df["target_pos_y"].values.copy()
    actual = df["eef_pos_y"].values.copy()

    target -= np.mean(target)
    actual -= np.mean(actual)

    corr = np.correlate(actual, target, mode="full")

    lag_idx = corr.argmax() - (len(target) - 1)

    dt = np.mean(np.diff(df["t"]))

    lag_time = lag_idx * dt

    return lag_time


def analyze_frequency_sweep():

    rows = []

    for csv_file in sorted(glob.glob(os.path.join(DATA_DIR, "sine_y_freq_*.csv"))):

        freq = float(
            csv_file.split("_")[-1].replace(".csv", "")
        )

        df = pd.read_csv(csv_file)

        target = df["target_pos_y"].values
        actual = df["eef_pos_y"].values

        # amp_target = (
        #     np.max(target) - np.min(target)
        # ) / 2
        # amp_actual = (
        #     np.max(actual) - np.min(actual)
        # ) / 2

        # skip first 2 cycles to avoid transient inflation
        period = 1.0 / freq
        skip_time = 2 * period
        df_ss = df[df["t"] > skip_time]
        amp_target = (df_ss["target_pos_y"].max() - df_ss["target_pos_y"].min()) / 2
        amp_actual = (df_ss["eef_pos_y"].max() - df_ss["eef_pos_y"].min()) / 2

        gain = amp_actual / amp_target if amp_target > 0 else float("nan")

        lag_time = estimate_phase_lag(df)

        phase_deg = lag_time * freq * 360.0

        rms_error = rms(target - actual)

        rows.append({
            "frequency_hz": freq,
            "gain": gain,
            "phase_lag_deg": phase_deg,
            "rms_error_m": rms_error,
        })

    results = pd.DataFrame(rows)

    results.sort_values(
        "frequency_hz",
        inplace=True
    )

    results.to_csv(
        os.path.join(DATA_DIR, "frequency_sweep_summary.csv"),
        index=False
    )

    # Gain Plot
    plt.figure(figsize=(6,4))

    plt.plot(
        results["frequency_hz"],
        results["gain"],
        marker="o"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude Ratio")
    plt.title("Frequency Response")

    plt.grid(True)

    plt.savefig(
        os.path.join(FIGURES_DIR, "frequency_gain.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Phase Plot
    plt.figure(figsize=(6,4))

    plt.plot(
        results["frequency_hz"],
        results["phase_lag_deg"],
        marker="o"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase Lag (deg)")
    plt.title("Phase Lag vs Frequency")

    plt.grid(True)

    plt.savefig(
        os.path.join(FIGURES_DIR, "frequency_phase.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # RMS-error Plot
    plt.figure(figsize=(6,4))

    plt.plot(
        results["frequency_hz"],
        results["rms_error_m"],
        marker="o"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RMS Error (m)")
    plt.title("Tracking Error vs Frequency")

    plt.grid(True)

    plt.savefig(
        os.path.join(FIGURES_DIR, "frequency_rms_error.png"),
        dpi=300,
        bbox_inches="tight"
    )

    print(results)

    # === BW Est === #
    bandwidth = np.nan
    for i in range(len(results)-1):

        g1 = results["gain"].iloc[i]
        g2 = results["gain"].iloc[i+1]

        if g1 >= 0.707 and g2 <= 0.707:

            f1 = results["frequency_hz"].iloc[i]
            f2 = results["frequency_hz"].iloc[i+1]

            bandwidth = np.interp(
                0.707,
                [g2, g1],
                [f2, f1]
            )

            break

    print(
        f"\nEstimated bandwidth = "
        f"{bandwidth:.2f} Hz"
    )

# ============================================================
# DAMPING RATIO SWEEP
# ============================================================

# def analyze_damping_sweep():
#     rows = []
#     for csv_file in sorted(
#         glob.glob(os.path.join(DATA_DIR, "step_y_dampratio_*.csv"))
#     ):
#         ratio = float(
#             csv_file.split("_")[-1].replace(".csv", "")
#         )
#         df = pd.read_csv(csv_file)
#         rms_error = rms(df["err_norm"])
#         peak_error = np.max(df["err_norm"])
#         max_vel = np.max(df["eef_vel_norm"])
#         rows.append({
#             "damping_ratio": ratio,
#             "rms_error_m": rms_error,
#             "peak_error_m": peak_error,
#             "max_velocity_m_s": max_vel,
#         })
#     results = pd.DataFrame(rows)
#     results.sort_values(
#         "damping_ratio",
#         inplace=True
#     )
#     results.to_csv(
#         "damping_sweep_summary.csv",
#         index=False
#     )
#     plt.figure(figsize=(6,4))
#     plt.plot(
#         results["damping_ratio"],
#         results["rms_error_m"],
#         marker="o"
#     )
#     plt.xlabel("Damping Ratio")
#     plt.ylabel("RMS Error (m)")
#     plt.title("Tracking Error vs Damping Ratio")
#     plt.grid(True)
#     plt.savefig(
#         "figures/damping_ratio_vs_rms_error.png",
#         dpi=300,
#         bbox_inches="tight"
#     )
#     plt.close()
#     print(results)
def analyze_damping_sweep():

    files = sorted(glob.glob(os.path.join(DATA_DIR, "sine_y_dampratio_*.csv")))
    if not files:
        print("  [damping sine] no sine_y_dampratio_*.csv files found, skipping.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(files)))

    for i, csv_file in enumerate(files):
        ratio = float(csv_file.split("_")[-1].replace(".csv", ""))
        df    = pd.read_csv(csv_file)

        # use steady-state only — skip first 2 cycles at 1Hz = 2s
        df = df[df["t"] > 2.0].copy()
        t  = df["t"].values

        tgt = df["target_pos_y"].values - df["target_pos_y"].mean()
        act = df["eef_pos_y"].values    - df["eef_pos_y"].mean()

        # phase lag via cross-correlation
        corr    = np.correlate(act, tgt, mode="full")
        lag_idx = corr.argmax() - (len(tgt) - 1)
        dt      = np.mean(np.diff(t))
        lag_ms  = lag_idx * dt * 1000.0

        axes[0].plot(t, tgt * 1000, "--", color="gray", linewidth=0.8,
                     label="Target" if i == 0 else None)
        axes[0].plot(t, act * 1000, color=colors[i], linewidth=1.5,
                     label=f"ζ={ratio}  lag={lag_ms:.0f}ms")

        axes[1].plot(t, df["err_norm"].values * 1000,
                     color=colors[i], linewidth=1.2, label=f"ζ={ratio}")

    axes[0].set_ylabel("Position (mm, centred)")
    axes[0].set_title("Sine tracking vs damping ratio @ 0.5Hz\n(phase lag quantifies felt lagginess)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Tracking error |e| (mm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "damping_sine_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# KP SWEEP
# ============================================================
def analyze_kp_sweep():

    rows = []

    for csv_file in sorted(glob.glob(os.path.join(DATA_DIR, "sine_y_kp_*.csv"))):

        kp = float(
            csv_file.split("_")[-1].replace(".csv", "")
        )

        df = pd.read_csv(csv_file)

        rms_error = rms(df["err_norm"])

        peak_error = np.max(df["err_norm"])

        rows.append({
            "kp": kp,
            "rms_error_m": rms_error,
            "peak_error_m": peak_error,
        })

    results = pd.DataFrame(rows)

    results.sort_values("kp", inplace=True)

    results.to_csv(
        os.path.join(DATA_DIR, "kp_metrics.csv"),
        index=False
    )

    plt.figure()

    plt.plot(
        results["kp"],
        results["rms_error_m"],
        marker="o"
    )

    plt.xlabel("Kp")
    plt.ylabel("RMS Error (m)")
    plt.title("Tracking Error vs Kp")
    plt.grid()

    plt.savefig(
        os.path.join(FIGURES_DIR, "kp_vs_rms_error.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(results)


# ============================================================
# SIM_HZ SWEEP
# ============================================================
def analyze_simhz_sweep():

    rows = []

    for csv_file in sorted(
        glob.glob(os.path.join(DATA_DIR, "sine_y_simhz_*.csv"))
    ):

        simhz = float(
            csv_file.split("_")[-1].replace(".csv", "")
        )

        df = pd.read_csv(csv_file)

        rms_error = rms(df["err_norm"])

        peak_error = np.max(df["err_norm"])

        rows.append({
            "sim_hz": simhz,
            "rms_error_m": rms_error,
            "peak_error_m": peak_error,
        })

    results = pd.DataFrame(rows)

    results.sort_values("sim_hz", inplace=True)

    results.to_csv(
        os.path.join(DATA_DIR, "simhz_metrics.csv"),
        index=False
    )

    plt.figure()

    plt.plot(
        results["sim_hz"],
        results["rms_error_m"],
        marker="o"
    )

    plt.xlabel("Simulation Frequency (Hz)")
    plt.ylabel("RMS Error (m)")
    plt.title("Tracking Error vs Simulation Frequency [Hz]")
    plt.grid()

    plt.savefig(
        os.path.join(FIGURES_DIR, "simhz_vs_rms_error.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(results)

# ============================================================
# FORCE UPDATE LATENCY (ZOH) vs SIM_HZ
# ============================================================

def analyze_force_latency():
    """
    For each sine_y_simhz_*.csv, compute the actual inter-step dt distribution
    from the logged timestamps. Worst-case ZOH hold period = max(dt) per run.
    Also overlays the theoretical 1/SIM_HZ hold period for reference.

    Produces:
      figures/force_latency_distribution.png  — violin/box of dt per sim rate
      figures/force_latency_zoh_vs_simhz.png  — mean/max dt vs sim rate
    """

    files = sorted(glob.glob(os.path.join(DATA_DIR, "sine_y_simhz_*.csv")))
    if not files:
        print("  [force latency] no sine_y_simhz_*.csv files found, skipping.")
        return

    all_dts   = {}
    sim_rates = []

    for csv_file in files:
        simhz = float(csv_file.split("_")[-1].replace(".csv", ""))
        df    = pd.read_csv(csv_file)
        dts   = np.diff(df["t"].values) * 1000.0  # ms
        dts   = dts[dts > 0]                       # drop any zero-diff artefacts
        all_dts[simhz]  = dts
        sim_rates.append(simhz)

    sim_rates = sorted(sim_rates)

    # ── violin plot of dt distributions ──
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = range(len(sim_rates))
    vp = ax.violinplot(
        [all_dts[hz] for hz in sim_rates],
        positions=positions,
        showmedians=True,
        showextrema=True,
    )
    for body in vp["bodies"]:
        body.set_alpha(0.6)

    # overlay theoretical 1/SIM_HZ hold period
    for i, hz in enumerate(sim_rates):
        ax.axhline(
            y=1000.0 / hz,
            xmin=(i - 0.35) / len(sim_rates),
            xmax=(i + 0.35) / len(sim_rates),
            color="red",
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{int(hz)} Hz" for hz in sim_rates])
    ax.set_xlabel("Simulation frequency")
    ax.set_ylabel("Step dt (ms)")
    ax.set_title("Force update latency distribution per sim rate\n(red dashed = theoretical 1/SIM_HZ)")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "force_latency_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── mean / max dt vs sim rate ──
    means    = [np.mean(all_dts[hz])  for hz in sim_rates]
    maxes    = [np.max(all_dts[hz])   for hz in sim_rates]
    theoret  = [1000.0 / hz           for hz in sim_rates]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sim_rates, theoret, "k--",  marker="",  label="Theoretical 1/SIM_HZ")
    ax.plot(sim_rates, means,   "o-",               label="Mean dt")
    ax.plot(sim_rates, maxes,   "s--",  alpha=0.7,  label="Max dt (worst-case ZOH)")
    ax.set_xlabel("Simulation frequency (Hz)")
    ax.set_ylabel("dt (ms)")
    ax.set_title("ZOH force-update hold period vs sim rate")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "force_latency_zoh_vs_simhz.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("  [force latency]")
    for hz in sim_rates:
        d = all_dts[hz]
        print(f"    {int(hz):3d} Hz  mean={np.mean(d):.2f}ms  "
              f"std={np.std(d):.2f}ms  max={np.max(d):.2f}ms  "
              f"theoretical={1000/hz:.2f}ms")


# ============================================================
# PHASE LAG WITH 90-DEG DESTABILISATION LINE
# ============================================================

def plot_phase_lag_with_limit():
    """
    Re-plots the phase-lag vs frequency curve from frequency_sweep_summary.csv
    (produced by analyze_frequency_sweep) and overlays a 90° hard-limit line,
    annotating the estimated crossover frequency where haptic feel degrades.

    Produces:
      figures/phase_lag_teleop_limit.png
    """

    summary_file = "frequency_sweep_summary.csv"
    if not os.path.exists(summary_file):
        print("  [phase lag limit] frequency_sweep_summary.csv not found; "
              "run analyze_frequency_sweep() first.")
        return

    results = pd.read_csv(summary_file)
    results.sort_values("frequency_hz", inplace=True)

    freqs  = results["frequency_hz"].values
    phases = results["phase_lag_deg"].values

    # interpolate 90-deg crossover
    crossover_hz = np.nan
    for i in range(len(phases) - 1):
        if phases[i] <= 90.0 <= phases[i + 1]:
            crossover_hz = np.interp(90.0, [phases[i], phases[i + 1]],
                                           [freqs[i],  freqs[i + 1]])
            break
        elif phases[i] >= 90.0 and i == 0:
            crossover_hz = freqs[0]
            break

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, phases, "o-", label="Phase lag")
    ax.axhline(y=90.0,  color="red",    linestyle="--", linewidth=1.5,
               label="90° limit (haptic destabilisation risk)")
    ax.axhline(y=180.0, color="orange", linestyle=":",  linewidth=1.2,
               label="180° (phase reversal)")

    if not np.isnan(crossover_hz):
        ax.axvline(x=crossover_hz, color="red", linestyle="--", alpha=0.4)
        ax.text(crossover_hz + 0.05, 95,
                f"≈{crossover_hz:.2f} Hz",
                color="red", fontsize=9)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase lag (deg)")
    ax.set_title("Phase lag vs frequency — teleop feel limits")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase_lag_teleop_limit.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [phase lag limit] 90° crossover ≈ {crossover_hz:.2f} Hz")


# ============================================================
# EEF VELOCITY PROFILE DURING STEP RESPONSE
# ============================================================

def plot_step_velocity_profiles():
    """
    For each step_response_*.csv, plots the EEF velocity magnitude alongside
    position tracking on a twin-axis figure, showing whether the arm saturates
    the OSC velocity output during the overshoot.

    Produces:
      figures/step_velocity_<axis>.png  for each axis found
    """

    files = sorted(glob.glob(os.path.join(DATA_DIR, "step_response_*.csv")))
    if not files:
        print("  [step velocity] no step_response_*.csv files found, skipping.")
        return

    for csv_file in files:
        df   = pd.read_csv(csv_file)
        axis = get_axis_from_filename(csv_file)
        t    = df["t"].values

        target = df[f"target_pos_{axis}"].values
        actual = df[f"eef_pos_{axis}"].values
        vel    = df["eef_vel_norm"].values

        # find step time
        step_idxs = np.where(np.abs(np.diff(target)) > 1e-6)[0]
        if len(step_idxs) == 0:
            continue
        t_step = t[step_idxs[0]]

        # zoom window: 0.5s before step to 3s after
        mask = (t >= t_step - 0.5) & (t <= t_step + 3.0)
        t_w, tgt_w, act_w, vel_w = t[mask], target[mask], actual[mask], vel[mask]

        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax1.plot(t_w, tgt_w, "--", color="steelblue",  label="Target pos", linewidth=1.2)
        ax1.plot(t_w, act_w, "-",  color="steelblue",  label="EEF pos",    linewidth=1.5)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")

        ax2 = ax1.twinx()
        ax2.fill_between(t_w, vel_w, alpha=0.25, color="darkorange")
        ax2.plot(t_w, vel_w, color="darkorange", linewidth=1.2, label="EEF speed")
        ax2.set_ylabel("|EEF velocity| (m/s)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")

        # peak velocity annotation
        peak_vel     = np.max(vel_w)
        peak_vel_idx = np.argmax(vel_w)
        ax2.annotate(
            f"peak {peak_vel:.3f} m/s",
            xy=(t_w[peak_vel_idx], peak_vel),
            xytext=(t_w[peak_vel_idx] + 0.1, peak_vel * 0.85),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="darkorange"),
            color="darkorange",
        )

        ax1.axvline(x=t_step, color="gray", linestyle=":", linewidth=1, label="Step")

        # combined legend
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

        ax1.set_title(f"EEF velocity during step response — {axis.upper()} axis")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()

        save_name = f"step_velocity_{axis}.png"
        fig.savefig(os.path.join(FIGURES_DIR, save_name), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [step velocity] {axis.upper()}: peak speed = {peak_vel:.3f} m/s")


# ============================================================
# CROSS-AXIS COUPLING
# ============================================================

def plot_cross_axis_coupling():
    """
    For each step_response_*.csv, plots the off-axis position deviations
    (relative to their pre-step values) during the step.  Relevant for
    peg-in-hole: any Y/Z drift during an X step (or X/Y during a Z step)
    directly adds to insertion misalignment.

    Produces:
      figures/cross_axis_coupling_<axis>.png  for each axis found
    """

    files = sorted(glob.glob(os.path.join(DATA_DIR, "step_response_*.csv")))
    if not files:
        print("  [cross-axis] no step_response_*.csv files found, skipping.")
        return

    for csv_file in files:
        df   = pd.read_csv(csv_file)
        axis = get_axis_from_filename(csv_file)
        t    = df["t"].values

        target = df[f"target_pos_{axis}"].values
        step_idxs = np.where(np.abs(np.diff(target)) > 1e-6)[0]
        if len(step_idxs) == 0:
            continue
        t_step    = t[step_idxs[0]]
        step_idx  = step_idxs[0] + 1

        off_axes = [a for a in ("x", "y", "z") if a != axis]

        # baseline = mean of first 50 pre-step samples (or up to step_idx)
        pre = slice(max(0, step_idx - 50), step_idx)

        fig, axes_list = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

        # top: primary axis tracking (context)
        ax_main = axes_list[0]
        baseline_main = np.mean(df[f"eef_pos_{axis}"].values[pre])
        ax_main.plot(t, target - target[0],
                     "--", color="steelblue", linewidth=1.2, label="Target (shifted)")
        ax_main.plot(t, df[f"eef_pos_{axis}"].values - baseline_main,
                     color="steelblue", linewidth=1.5, label=f"EEF {axis.upper()} (shifted)")
        ax_main.axvline(x=t_step, color="gray", linestyle=":", linewidth=1)
        ax_main.set_ylabel("Position (m)")
        ax_main.set_title(f"Cross-axis coupling — step in {axis.upper()}")
        ax_main.legend(fontsize=8)
        ax_main.grid(True, alpha=0.3)

        # bottom: off-axis drift (deviation from baseline)
        ax_off = axes_list[1]
        colors = ["darkorange", "green"]
        for oa, col in zip(off_axes, colors):
            baseline_oa = np.mean(df[f"eef_pos_{oa}"].values[pre])
            drift = (df[f"eef_pos_{oa}"].values - baseline_oa) * 1000.0  # mm
            peak_drift = np.max(np.abs(drift[step_idx:]))
            ax_off.plot(t, drift, color=col, linewidth=1.2,
                        label=f"{oa.upper()} drift  (peak {peak_drift:.2f} mm)")

        ax_off.axvline(x=t_step, color="gray", linestyle=":", linewidth=1)
        ax_off.axhline(y=0, color="black", linewidth=0.6, linestyle="-")
        ax_off.set_xlabel("Time (s)")
        ax_off.set_ylabel("Off-axis drift (mm)")
        ax_off.legend(fontsize=8)
        ax_off.grid(True, alpha=0.3)

        fig.tight_layout()
        save_name = f"cross_axis_coupling_{axis}.png"
        fig.savefig(os.path.join(FIGURES_DIR, save_name), dpi=300, bbox_inches="tight")
        plt.close(fig)

        for oa in off_axes:
            baseline_oa = np.mean(df[f"eef_pos_{oa}"].values[pre])
            drift = (df[f"eef_pos_{oa}"].values[step_idx:] - baseline_oa) * 1000.0
            print(f"  [cross-axis] step {axis.upper()}  off-axis {oa.upper()}: "
                  f"peak drift = {np.max(np.abs(drift)):.3f} mm")


# ============================================================
# TRACKING ERROR TIME-SERIES: 0.5 Hz vs 5 Hz OVERLAID
# ============================================================

def plot_bandwidth_comparison():
    """
    Overlays the tracking error time-series for the lowest and highest
    frequency sine runs to make the bandwidth limitation visually obvious.
    Uses sine_y_simhz_100.csv (0.5 Hz) and sine_y_freq_5_0.csv (5 Hz).
    Falls back gracefully if either file is missing.

    Also plots target vs EEF position for each in a 2-row panel.

    Produces:
      figures/bandwidth_comparison_error.png
      figures/bandwidth_comparison_tracking.png
    """

    # find lowest-freq sine file (prefer simhz_100 = 0.5 Hz, else first freq file)
    low_file  = "sine_y_simhz_100.csv"
    high_file = "sine_y_freq_5_0.csv"

    missing = [f for f in (low_file, high_file) if not os.path.exists(f)]
    if missing:
        print(f"  [bandwidth comparison] missing files: {missing}, skipping.")
        return

    df_low  = pd.read_csv(low_file)
    df_high = pd.read_csv(high_file)

    low_label  = "0.5 Hz sine"
    high_label = "5.0 Hz sine"

    # ── error magnitude comparison ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_low["t"],  df_low["err_norm"]  * 1000,
            color="steelblue",  linewidth=0.9, alpha=0.85, label=low_label)
    ax.plot(df_high["t"], df_high["err_norm"] * 1000,
            color="darkorange", linewidth=0.9, alpha=0.85, label=high_label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking error |e| (mm)")
    ax.set_title("Tracking error: 0.5 Hz vs 5 Hz sine — bandwidth impact")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "bandwidth_comparison_error.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── target vs EEF tracking (2-row panel) ──
    fig, (ax_l, ax_h) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # centre both signals at zero for easy visual comparison
    for ax, df, freq_label, col in [
        (ax_l, df_low,  low_label,  "steelblue"),
        (ax_h, df_high, high_label, "darkorange"),
    ]:
        tgt = df["target_pos_y"].values
        act = df["eef_pos_y"].values
        tgt = tgt - np.mean(tgt)
        act = act - np.mean(act)
        ax.plot(df["t"], tgt * 1000, "--", color=col, linewidth=1.2,
                alpha=0.7, label="Target")
        ax.plot(df["t"], act * 1000, "-",  color=col, linewidth=1.5,
                label="EEF")
        ax.set_ylabel("Position (mm, centred)")
        ax.set_title(freq_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_h.set_xlabel("Time (s)")
    fig.suptitle("Target vs EEF tracking: 0.5 Hz vs 5 Hz", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "bandwidth_comparison_tracking.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    rms_low  = rms(df_low["err_norm"].values)  * 1000
    rms_high = rms(df_high["err_norm"].values) * 1000
    print(f"  [bandwidth comparison] RMS error — {low_label}: {rms_low:.2f} mm  |  "
          f"{high_label}: {rms_high:.2f} mm")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n=== STEP RESPONSE ANALYSIS ===")
    analyze_step_responses()

    print("\n=== FREQUENCY SWEEP ANALYSIS ===")
    analyze_frequency_sweep()

    print("\n=== DAMPING RATIO SWEEP ANALYSIS ===")
    analyze_damping_sweep()

    print("\n=== KP SWEEP ANALYSIS ===")
    analyze_kp_sweep()

    print("\n=== SIM_HZ SWEEP ANALYSIS ===")
    analyze_simhz_sweep()

    print("\n=== FORCE UPDATE LATENCY (ZOH) ===")
    analyze_force_latency()

    print("\n=== PHASE LAG WITH TELEOP DESTABILISATION LIMIT ===")
    plot_phase_lag_with_limit()

    print("\n=== EEF VELOCITY DURING STEP RESPONSE ===")
    plot_step_velocity_profiles()

    print("\n=== CROSS-AXIS COUPLING ===")
    plot_cross_axis_coupling()

    print("\n=== BANDWIDTH COMPARISON (0.5 Hz vs 5 Hz) ===")
    plot_bandwidth_comparison()

    print("\nDone.")