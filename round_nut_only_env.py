import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T


class StaticRoundNutObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/round-nut.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle_site"})
        return dic


class RoundNutOnlyEnv(ManipulationEnv):
    """
    Minimal round-nut-only task using a plain table arena.

    The scene contains only:
      - a single round nut
      - a table
      - the robot

    The nut is initialized at a fixed position above the table.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.07), #0.05
        table_friction=(2.0, 0.02, 0.001),
        table_offset=(0, 0, 0.9), #0.82
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
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale

        self.round_nut = StaticRoundNutObject(name="RoundNut")
        self.round_nut_start_pos = np.array([0.0, 0.0, table_offset[2] + 0.02])
        self.round_nut.set_pos(self.round_nut_start_pos)

        self.cylinder = CylinderObject(
            name="InsertCylinder",
            size=(0.0275, 0.0325),
            friction=(3.4, 0.3, 0.01), # sliding, torsional, rolling
            # rgba=[0.2, 0.7, 1.0, 1.0], # rich blue color to visually distinguish the cylinder from the round nut
            rgba=[0.75, 0.75, 0.75, 1.0], # light grey color to match the round nut, so that it is harder to visually distinguish the two objects and you rely on haptics more
        )
        self.cylinder_start_pos = np.array([0.12, 0.12, table_offset[2] + 0.02])
        # self.cylinder.set_pos(self.cylinder_start_pos)

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
        return 0.0

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            has_legs=False,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.round_nut, self.cylinder],
        )

    def _setup_references(self):
        super()._setup_references()
        self.table_body_id = self.sim.model.body_name2id("table")
        self.round_nut_body_id = self.sim.model.body_name2id("RoundNut_main")
        self.cylinder_body_id = self.sim.model.body_name2id("InsertCylinder_main")

        cylinder_joint_id = self.sim.model.joint_name2id("InsertCylinder_joint0")
        self.cylinder_qpos_adr = self.sim.model.jnt_qposadr[cylinder_joint_id]
        self.cylinder_dof_adr = self.sim.model.jnt_dofadr[cylinder_joint_id]

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def round_nut_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.round_nut_body_id])

            @sensor(modality=modality)
            def round_nut_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.round_nut_body_id], to="xyzw")

            @sensor(modality=modality)
            def cylinder_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cylinder_body_id])

            @sensor(modality=modality)
            def cylinder_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.cylinder_body_id], to="xyzw")

            for s in [round_nut_pos, round_nut_quat, cylinder_pos, cylinder_quat]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.qpos[self.cylinder_qpos_adr:self.cylinder_qpos_adr + 3] = self.cylinder_start_pos
        self.sim.data.qpos[self.cylinder_qpos_adr + 3:self.cylinder_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.qvel[self.cylinder_dof_adr:self.cylinder_dof_adr + 6] = 0
