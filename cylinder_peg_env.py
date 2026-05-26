"""
CylinderArena and PegInHoleEnv
================================
Custom robosuite arena and environment for haptic peg-in-hole task.

Scene:
  - hole  : cylinder welded to table (radius 22mm, depth 60mm) — orange
  - cylinder_peg : free cylinder (radius 18mm, length 120mm)   — blue, graspable

Usage:
    from cylinder_peg_env import PegInHoleEnv
    env = suite.make("PegInHole", ...)   # after registering, or instantiate directly
"""

import os
import numpy as np

from robosuite.models.arenas import Arena
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T

import robosuite
# Place cylinder_arena.xml alongside robosuite's other arena XMLs
# so relative texture paths (../textures/) resolve correctly
ROBOSUITE_ARENAS = os.path.join(
    os.path.dirname(robosuite.__file__),
    "models", "assets", "arenas"
)
ARENA_XML = os.path.join(ROBOSUITE_ARENAS, "cylinder_arena.xml")


# ─────────────────────────────────────────────────────────────
# ARENA
# ─────────────────────────────────────────────────────────────

class CylinderArena(Arena):
    """
    Arena with a welded hole cylinder and a free peg cylinder.
    Loaded from cylinder_arena.xml in the same directory.
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.82),
    ):
        self.table_full_size  = np.array(table_full_size)
        self.table_friction   = table_friction
        self.table_offset     = table_offset
        super().__init__(ARENA_XML)

    def set_origin(self, origin):
        """Move entire arena by offset."""
        offset = np.array(origin)
        node = self.worldbody
        for body in node.findall("body"):
            pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            body.set("pos", " ".join(f"{v:.4f}" for v in pos + offset))


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class PegInHoleEnv(ManipulationEnv):
    """
    Peg-in-hole task using CylinderArena.

    The robot picks up the blue cylinder peg and inserts it
    into the orange hole welded to the table.

    Key bodies:
        cylinder_peg  — free cylinder, graspable
        hole          — welded cylinder, fixed to table
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.82),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=100000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction  = table_friction
        self.table_offset    = table_offset
        self.use_object_obs  = use_object_obs
        self.reward_scale    = reward_scale

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Sparse reward: 1.0 if peg is inside hole, else 0.
        """
        peg_pos  = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        dist_xy  = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
        in_hole  = dist_xy < 0.005 and peg_pos[2] < hole_pos[2] + 0.05
        reward   = 1.0 if in_hole else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _load_model(self):
        super()._load_model()

        # Position robot relative to table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Load our custom arena
        mujoco_arena = CylinderArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Build task with no extra objects (peg and hole are in the arena XML)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )

    def _setup_references(self):
        super()._setup_references()
        self.table_body_id = self.sim.model.body_name2id("table")
        self.peg_body_id   = self.sim.model.body_name2id("cylinder_peg")
        self.hole_body_id  = self.sim.model.body_name2id("hole")
        # qpos address for the peg freejoint
        jnt_id = self.sim.model.joint_name2id("cylinder_peg_joint")
        self.peg_qpos_adr = self.sim.model.jnt_qposadr[jnt_id]
        self.peg_dof_adr  = self.sim.model.jnt_dofadr[jnt_id]

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg_body_id])

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")

            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hole_body_id])

            for s in [peg_pos, peg_quat, hole_pos]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        # Place peg upright on table next to hole at reset
        # hole is at [0.10, 0.0, 0.85] — place peg at [0.10, 0.12, 0.87]
        peg_start_pos  = np.array([0.10, 0.12, 0.87])
        peg_start_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        self.sim.data.qpos[self.peg_qpos_adr:self.peg_qpos_adr+3] = peg_start_pos
        self.sim.data.qpos[self.peg_qpos_adr+3:self.peg_qpos_adr+7] = peg_start_quat
        self.sim.data.qvel[self.peg_dof_adr:self.peg_dof_adr+6] = 0

    def _check_success(self):
        peg_pos  = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        dist_xy  = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
        return bool(dist_xy < 0.005 and peg_pos[2] < hole_pos[2] + 0.05)