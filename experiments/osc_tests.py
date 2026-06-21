# step_response_test.py
import json
import time
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

import robosuite as suite
from round_nut_only_env import RoundNutOnlyEnv
from analyze_osc_results import (
    analyze_step_responses,
    analyze_frequency_sweep,
    analyze_damping_sweep,
    analyze_kp_sweep,
    analyze_simhz_sweep,
)

def apply_solref_patches(env):
    # solref = [timeconst, dampratio]
    #   timeconst : smaller -> stiffer surface (0.001=hard metal, 0.02=soft)
    #   dampratio : 1.0=critically damped (no bounce), >1.0=overdamped
    for i in range(env.sim.model.ngeom):
        name = env.sim.model.geom_id2name(i)
        if 'table' in name.lower():
            env.sim.model.geom_solref[i] = [0.02, 2.0]
        elif any(g in name for g in ['finger', 'hand_collision']):
            env.sim.model.geom_solref[i] = [0.01, 2.0]

def run_osc_experiments(
          controller_config,
          target_generator,
          csv_name,
          duration=10.0,
          step_time=2.0,
          sim_hz=100
):
    """
    Step Reponse Test for OSC control of Panda arm
    ======================================
    Obtain step response data for OSC control of Panda EEF in robosuite.
    No Touch input needed, target position is set to a step function (20cm in X direction) at STEP_TIME=2s, and the robot's response is recorded for 10 seconds.
    Measures the target position (target), actual position (eef_pos), position tracking error (pos_error).

    Data is saved as a csv file with columns: time, target_x, target_y, target_z, eef_x, eef_y, eef_z, err_x, err_y, err_z.
    
    INPUTS:
    - target_generator: function that generates the target position given time t (e.g. step function)
    - csv_name: string, name of the output csv file (e.g. "step_response_data.csv")
    - duration: total duration of the test in seconds (default 10.0)
    - step_time: time at which the step change occurs in seconds (default 2.
    
    """

    # ----------------------------
    # PARAMETERS 
    # ----------------------------

    ACTION_GAIN = 5.0 # same as in robosuite_haptic_bridge.py

    CSV_FILENAME = os.path.join(DATA_DIR, f"{csv_name}.csv")

    SIM_HZ = sim_hz
    STEP_TIME = step_time
    DURATION = duration

    # STEP_SIZE = np.array([0.2, 0, 0]) # Step size in meters (20cm in X direction)
    
    # ----------------------------
    # LOAD CONTROLLER & ENVIRONMENT
    # ----------------------------

    env = RoundNutOnlyEnv(
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=True,
            control_freq=SIM_HZ,
            horizon=100000,
        )
    env.robots[0].init_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    obs = env.reset()

    apply_solref_patches(env)

    for _ in range(50):
            action = np.zeros(env.action_dim)
            action[-1] = 1.0
            obs, _, _, _ = env.step(action)

    eef_start = obs['robot0_eef_pos'].copy()
    print(f"  EEF start    : {eef_start.round(3)}")
    print("  Environment ready.")

    # ----------------------------
    # LOGGER 
    # ----------------------------
    log = []

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    dt = 1.0 / SIM_HZ # Simulation timestep
    num_steps = int(DURATION / dt)

    for k in range(num_steps):
        t = k * dt

        # Determine target position (step function)
        target_pos = target_generator(t, eef_start)

        eef_pos = obs['robot0_eef_pos']
        pos_error = target_pos - eef_pos
        err_norm = np.linalg.norm(pos_error)
        # Compute action to move towards target position
        action = np.zeros(env.action_dim)
        action[:3] = np.clip(pos_error * ACTION_GAIN, -1.0, 1.0)
        action[-1] = 1.0 # Gripper open

        obs, _, _, _ = env.step(action)

        env.render()

        eef_vel = env.sim.data.get_body_xvelp("robot0_right_hand")

    # ----------------------------
    # LOGGING AND DATA SAVING
    # ----------------------------
        log.append({
            "t": t,
            "target_pos_x": target_pos[0],
            "target_pos_y": target_pos[1],
            "target_pos_z": target_pos[2],
            "eef_pos_x": eef_pos[0],
            "eef_pos_y": eef_pos[1],
            "eef_pos_z": eef_pos[2],
            "err_x": pos_error[0],
            "err_y": pos_error[1],
            "err_z": pos_error[2],
            "err_norm": err_norm,
            "action_x": action[0],
            "action_y": action[1],
            "action_z": action[2],
            "eef_vel_x": eef_vel[0],
            "eef_vel_y": eef_vel[1],
            "eef_vel_z": eef_vel[2],
            "eef_vel_norm": np.linalg.norm(eef_vel),
            })
        

    df = pd.DataFrame(log)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"  Step response data saved to {CSV_FILENAME}")
    env.close()


def step_x_generator(t, home_position, step_time=2.0, step_size=0.15):
    if t < step_time:
        return home_position
    else:
        return home_position + np.array([step_size, 0, 0])

def step_y_generator(t, home_position, step_time=2.0, step_size=0.2):
    if t < step_time:
        return home_position
    else:
        return home_position + np.array([0, step_size, 0])

def step_z_generator(t, home_position, step_time=2.0, step_size=0.2):
    if t < step_time:
        return home_position
    else:
        return home_position + np.array([0, 0, step_size])
        
def sine_x_generator(t, home_position, amplitude=0.05, frequency=1.0):
        return home_position + np.array([amplitude * np.sin(2 * np.pi * frequency * t), 0, 0])

def sine_y_generator(t, home_position, amplitude=0.05, frequency=0.5):
        return home_position + np.array([0, amplitude * np.sin(2 * np.pi * frequency * t), 0])
    
# def sine_wave_generator(t, home_position, axis, amplitude=0.05, frequency=1.0):
#     if axis == 'x':
#         return home_position + np.array([amplitude * np.sin(2 * np.pi * frequency * t), 0, 0])
#     elif axis == 'y':
#         return home_position + np.array([0, amplitude * np.sin(2 * np.pi * frequency * t), 0])
#     elif axis == 'z':
#         return home_position + np.array([0, 0, amplitude * np.sin(2 * np.pi * frequency * t)])

def reset_ctrl_config():
    ctrl_config_path = "osc_world_frame.json"
    with open(ctrl_config_path) as f:
        ctrl_config = json.load(f)
    print("  Using world frame OSC_POSE controller")
    return ctrl_config


if __name__ == "__main__":
    # ============================================================
    # EXPERIMENT SWITCHES
    # ============================================================
    RUN_STEP_RESPONSE = True
    RUN_FREQUENCY_SWEEP = True
    RUN_DAMPING_SWEEP = True
    RUN_KP_SWEEP = True
    RUN_SIMHZ_SWEEP = True
    RUN_ANALYSIS = True

    ctrl_config = reset_ctrl_config()

    # Example usage: run step response test with step function in X direction
    # === Step response in X direction ===
    if RUN_STEP_RESPONSE:
        print(" ==== STARTING STEP X RESPONSE ==== ")
        ctrl_config = reset_ctrl_config()
        run_osc_experiments(
            ctrl_config,
            step_x_generator,
            csv_name="step_response_x"
        )

        print(" ==== STARTING STEP Y RESPONSE ==== ")
        run_osc_experiments(
            ctrl_config,
            step_y_generator,
            csv_name="step_response_y"
        )
        print(" ==== STARTING STEP Z RESPONSE ==== ")
        run_osc_experiments(
            ctrl_config,
            step_z_generator,
            csv_name="step_response_z"
        )

    # === Frequency Sweep Sine y direction ===
    if RUN_FREQUENCY_SWEEP:
        ctrl_config = reset_ctrl_config()
        for frequency in [0.5, 1.0, 2.0, 3.0, 5.0]:
            print(
                f"==== STARTING FREQ SWEEP {frequency} Hz ===="
            )
            sine_gen = lambda t, home_pos: sine_y_generator(
                t,
                home_pos,
                amplitude=0.05,
                frequency=frequency,
            )
            run_osc_experiments(
                ctrl_config,
                sine_gen,
                csv_name=f"sine_y_freq_{frequency}",
                duration=20.0,
            )

    
    # === Damping Ratio Sweep Step y direction ===
    if RUN_DAMPING_SWEEP:
        ctrl_config = reset_ctrl_config()
        for dampratio in [0.2, 0.5, 1.0, 2.0]:
            print(
                f"==== STARTING DAMPING SWEEP {dampratio} ===="
            )

            ctrl_config["damping_ratio"] = dampratio
            run_osc_experiments(ctrl_config,
                lambda t, home: sine_y_generator(t, home, amplitude=0.05, frequency=0.5),
                csv_name=f"sine_y_dampratio_{dampratio}",
                duration=10.0,
            )

            # # === Damping Ratio Sweep Sine y direction ===
            # for dampratio in [0.2, 0.5, 1.0, 2.0]:
            #      print(f"  ==== STARTING DAMPRATIO SWEEP {dampratio} ==== ")
            #      ctrl_config["damping_ratio"] = dampratio
            #      run_osc_experiments(ctrl_config,
            #          sine_y_generator,
            #         csv_name="sine_y_dampratio_" + str(dampratio),
            #         duration=20.0,
            #      )


    # === Kp Sweep Sine y direction ===
    if RUN_KP_SWEEP:
        ctrl_config = reset_ctrl_config()
        for kp in [50, 100, 150, 200, 300]:
            print(
                f"==== STARTING KP SWEEP {kp} ===="
            )
            ctrl_config["kp"] = kp
            run_osc_experiments(
                ctrl_config,
                sine_y_generator,
                csv_name=f"sine_y_kp_{kp}",
                duration=20.0,
        )

    # === SIM_HZ Sweep Sine y direction ===
    ctrl_config = reset_ctrl_config()
    for simhz in [10, 20, 50, 100]:
         print(f"  ==== STARTING SIM_HZ SWEEP {simhz}Hz ==== ")
         run_osc_experiments(ctrl_config,
          sine_y_generator,
          csv_name="sine_y_simhz_" + str(simhz),
          duration=20.0,
          sim_hz=simhz
          )

    if RUN_ANALYSIS:
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

    print("\nDone.")