"""
TaskSorting environment — compatible with robosuite 1.5.1 and robomimic 0.5.0.

Inherits from ManipulationEnv (robosuite 1.5 replacement for SingleArmEnv).
Robomimic requires:
  - _check_success() -> bool or {"task": bool}
  - reward() -> float
  - _setup_observables() -> OrderedDict of Observable objects
  - _reset_internal() that sets joint positions via self.sim.data.set_joint_qpos
"""

from collections import OrderedDict

import numpy as np

from copy import copy

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import CylinderObject, BoxObject, PotWithHandlesObject
from robosuite.models.objects.composite.bin import Bin
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat, quat_multiply
from robosuite.utils.errors import RandomizationError

from .custom_arena import CustomArena


class SpacedUniformRandomSampler(UniformRandomSampler):
    """
    Subclass of UniformRandomSampler that enforces a minimum clearance margin
    between sampled objects (and already placed objects/fixtures) so that
    objects are not placed too close to each other, leaving sufficient clearance
    for the robot gripper to grasp them cleanly without collisions.
    """
    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_range=(0, 0),
        y_range=(0, 0),
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
        min_distance=0.08,
    ):
        self.min_distance = min_distance
        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            x_range=x_range,
            y_range=y_range,
            rotation=rotation,
            rotation_axis=rotation_axis,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def sample(self, fixtures=None, reference=None, on_top=True):
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert reference in placed_objects, f"Invalid reference received: {reference}"
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert base_offset.shape[0] == 3, f"Invalid reference received: {base_offset}"

        for obj in self.mujoco_objects:
            assert obj.name not in placed_objects, f"Object '{obj.name}' has already been sampled!"

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for i in range(5000):
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if other_obj.name == "sorting_bin":
                            # Physical rectangular footprint of sorting bin (0.30 x 0.30)
                            if abs(object_x - x) <= 0.16 and abs(object_y - y) <= 0.16:
                                location_valid = False
                                break
                        elif other_obj.name == "bowl":
                            # Physical radius of bowl/pot with handles (~0.16-0.18m)
                            if np.hypot(object_x - x, object_y - y) <= 0.18:
                                location_valid = False
                                break
                        else:
                            dist_xy = np.hypot(object_x - x, object_y - y)
                            req_dist = max(self.min_distance, other_obj.horizontal_radius + horizontal_radius)
                            if dist_xy <= req_dist and (
                                object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                            ):
                                location_valid = False
                                break

                if location_valid:
                    quat = self._sample_quat()
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError(f"Cannot place object {obj.name} in {self.name}")

        return placed_objects


class TaskSorting(ManipulationEnv):
    """
    Sorting task: the robot must pick red/blue cans and place them in the
    sorting bin.  Compatible with robosuite 1.5.1 and robomimic 0.5.0.

    Args:
        robots: robot specification forwarded to ManipulationEnv
        reward_shaping (bool): dense vs sparse reward
        reward_scale (float or None): scale the final reward
        use_object_obs (bool): include object state in observations
        placement_initializer: optional custom sampler
        **kwargs: all other kwargs forwarded to ManipulationEnv
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(1.2, 1.0, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        lift_target=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        # ── task-specific settings ───────────────────────────────────────────
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.custom_placement_initializer = placement_initializer
        self.placement_initializer = None
        self.lift_target = lift_target

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
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
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load XML model: arena, robot, objects."""
        super()._load_model()

        # ── Arena ────────────────────────────────────────────────────────────
        mujoco_arena = CustomArena(
            table_full_size=self.table_full_size,
            table_offset=tuple(self.table_offset),
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ── Robot base pose ──────────────────────────────────────────────────
        # Use the standard robosuite 1.5 pattern: offset from table size
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # ── Materials ────────────────────────────────────────────────────────
        tex_attrib = {"type": "cube"}

        red_mat = CustomMaterial(
            texture="SteelBrushed",
            tex_name="red_tex",
            mat_name="red_mat",
            tex_attrib=tex_attrib,
            mat_attrib={
                "texrepeat": "1 1",
                "specular": "0.9",
                "shininess": "0.95",
                "reflectance": "0.35",
                "rgba": "1 0 0 1",
            },
        )
        blue_mat = CustomMaterial(
            texture="SteelBrushed",
            tex_name="blue_tex",
            mat_name="blue_mat",
            tex_attrib=tex_attrib,
            mat_attrib={
                "texrepeat": "1 1",
                "specular": "0.9",
                "shininess": "0.95",
                "reflectance": "0.35",
                "rgba": "0 0 1 1",
            },
        )
        grey_mat = CustomMaterial(
            texture="SteelBrushed",
            tex_name="grey_tex",
            mat_name="grey_mat",
            tex_attrib=tex_attrib,
            mat_attrib={
                "texrepeat": "1 1",
                "specular": "0.9",
                "shininess": "0.95",
                "reflectance": "0.35",
                "rgba": "0.5 0.5 0.5 1",
            },
        )
        green_mat = CustomMaterial(
            texture="SteelBrushed",
            tex_name="green_tex",
            mat_name="green_mat",
            tex_attrib=tex_attrib,
            mat_attrib={
                "texrepeat": "1 1",
                "specular": "0.9",
                "shininess": "0.95",
                "reflectance": "0.35",
                "rgba": "0 1 0 1",
            },
        )

        # ── Objects ──────────────────────────────────────────────────────────
        self.red_can = CylinderObject(
            name="red_can",
            size_min=[0.02, 0.05],
            size_max=[0.02, 0.05],
            rgba=[1, 0, 0, 1],
            material=red_mat,
        )
        self.blue_can = CylinderObject(
            name="blue_can",
            size_min=[0.02, 0.05],
            size_max=[0.02, 0.05],
            rgba=[0, 0, 1, 1],
            material=blue_mat,
        )
        self.sorting_bin = Bin(
            name="sorting_bin",
            bin_size=(0.30, 0.30, 0.04),
            wall_thickness=0.01,
        )
        self.green_can = CylinderObject(
            name="green_can",
            size_min=[0.02, 0.05],
            size_max=[0.02, 0.05],
            rgba=[0, 1, 0, 1],
            material=green_mat,
        )
        self.bowl = PotWithHandlesObject(
            name="bowl",
            rgba_body=[1.0, 0.75, 0.0, 1.0],       # Bright yellow / amber body
            rgba_handle_0=[0.25, 0.25, 0.25, 1.0],  # Dark grey left handle
            rgba_handle_1=[0.25, 0.25, 0.25, 1.0],  # Dark grey right handle
            use_texture=False,                     # Use pure RGBA color without wood texture
        )
        self.yellow_cube = BoxObject(
            name="yellow_cube",
            size_min=[0.018, 0.018, 0.018],
            size_max=[0.018, 0.018, 0.018],
            rgba=[1.0, 1.0, 0.0, 1.0],  # Pure bright yellow for maximum VLM contrast
        )
        self.purple_cube = BoxObject(
            name="purple_cube",
            size_min=[0.018, 0.018, 0.018],
            size_max=[0.018, 0.018, 0.018],
            rgba=[0.5, 0.0, 0.5, 1.0],  # Deep vivid purple
        )

        self.objects = [
            self.red_can,
            self.blue_can,
            self.green_can,
            self.sorting_bin,
            self.bowl,
            self.yellow_cube,
            self.purple_cube,
        ]

        # ── Placement sampler ────────────────────────────────────────────────
        if self.custom_placement_initializer is not None:
            self.placement_initializer = self.custom_placement_initializer
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.objects)
        else:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            
            bin_sampler = UniformRandomSampler(
                name="BinSampler",
                mujoco_objects=[self.sorting_bin],
                x_range=[0.0, 0.0],       # Center X
                y_range=[0.18, 0.18],     # Moved closer to robot! (was 0.18)
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

            bowl_sampler = UniformRandomSampler(
                name="BowlSampler",
                mujoco_objects=[self.bowl],
                x_range=[-0.20, -0.18],   # Moved forward clear of robot base (was [-0.35, -0.28])
                y_range=[-0.30, -0.30],   # Shifted left clear of center swing (was -0.18)
                rotation=0.0,             # Keep standard symmetric handle orientation
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

            can_sampler = SpacedUniformRandomSampler(
                name="CanSampler",
                mujoco_objects=[
                    self.red_can,
                    self.blue_can,
                    self.green_can,
                ],
                x_range=[-0.02, 0.10],
                y_range=[-0.16, -0.04],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                min_distance=0.08,
            )

            cube_sampler = SpacedUniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=[
                    self.yellow_cube,
                    self.purple_cube,
                ],
                x_range=[-0.26, -0.14],
                y_range=[0.02, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                min_distance=0.08,
            )
            
            self.placement_initializer.append_sampler(bin_sampler)
            self.placement_initializer.append_sampler(bowl_sampler)
            self.placement_initializer.append_sampler(can_sampler)
            self.placement_initializer.append_sampler(cube_sampler)

        # ── Task tree ────────────────────────────────────────────────────────
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # References (called after sim is built)
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_references(self):
        """Grab MuJoCo body IDs for fast state look-ups."""
        super()._setup_references()
        self.red_can_body_id = self.sim.model.body_name2id(self.red_can.root_body)
        self.blue_can_body_id = self.sim.model.body_name2id(self.blue_can.root_body)
        self.green_can_body_id = self.sim.model.body_name2id(self.green_can.root_body)
        self.sorting_bin_id = self.sim.model.body_name2id(self.sorting_bin.root_body)
        self.bowl_id = self.sim.model.body_name2id(self.bowl.root_body)
        self.yellow_cube_body_id = self.sim.model.body_name2id(self.yellow_cube.root_body)
        self.purple_cube_body_id = self.sim.model.body_name2id(self.purple_cube.root_body)

    # ──────────────────────────────────────────────────────────────────────────
    # Observables  (robosuite 1.5 pattern — required by robomimic)
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_observables(self):
        """
        Sets up observables.  Mirrors the robosuite 1.5 Lift pattern so
        robomimic can find the expected observation keys.
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def red_can_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.red_can_body_id])

            @sensor(modality=modality)
            def red_can_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.red_can_body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def blue_can_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.blue_can_body_id])

            @sensor(modality=modality)
            def blue_can_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.blue_can_body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def green_can_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.green_can_body_id])

            @sensor(modality=modality)
            def green_can_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.green_can_body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def sorting_bin_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.sorting_bin_id])

            @sensor(modality=modality)
            def bowl_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bowl_id])

            @sensor(modality=modality)
            def yellow_cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.yellow_cube_body_id])

            @sensor(modality=modality)
            def yellow_cube_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.yellow_cube_body_id]), to="xyzw"
                )

            @sensor(modality=modality)
            def purple_cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.purple_cube_body_id])

            @sensor(modality=modality)
            def purple_cube_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.purple_cube_body_id]), to="xyzw"
                )

            sensors = [
                red_can_pos, red_can_quat,
                blue_can_pos, blue_can_quat,
                green_can_pos, green_can_quat,
                sorting_bin_pos, bowl_pos,
                yellow_cube_pos, yellow_cube_quat,
                purple_cube_pos, purple_cube_quat,
            ]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # ──────────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_internal(self):
        """Reset object poses via the placement sampler."""
        super()._reset_internal()

        if not self.deterministic_reset:
            from robosuite.utils.errors import RandomizationError
            success = False
            for _ in range(10):
                try:
                    object_placements = self.placement_initializer.sample()
                    success = True
                    break
                except RandomizationError:
                    continue
                    
            if not success:
                # Fallback to the last attempt, which will throw the error if it fails
                object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )
        
        # Pick a target for lift training. If explicitly set, use it; else pick randomly.
        if self.lift_target is not None:
            self.lift_target_name = self.lift_target
        else:
            targets = ["red_can", "blue_can", "green_can", "yellow_cube", "purple_cube"]
            self.lift_target_name = np.random.choice(targets)
            
        self.lift_target_obj = getattr(self, self.lift_target_name)
        self.lift_target_body_id = getattr(self, f"{self.lift_target_name}_body_id")

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    # ──────────────────────────────────────────────────────────────────────────
    # Reward  (required by robomimic via EnvRobosuite.get_reward)
    # ──────────────────────────────────────────────────────────────────────────

    def reward(self, action=None):
        """
        Sparse completion reward: +2.2 when the random target object is lifted above target height.
        With reward_shaping=True an additional reaching, grasping, and lifting reward is given.
        """
        reward = 0.0

        if self._check_success():
            reward = 2.2
        elif self.reward_shaping:
            eef_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])
            target_pos = np.array(self.sim.data.body_xpos[self.lift_target_body_id])
                
            # Reaching reward
            dist = np.linalg.norm(eef_pos - target_pos)
            reaching = 1.0 - np.tanh(10.0 * dist)
            reward += reaching
            
            # Grasping and lifting reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.lift_target_obj):
                reward += 0.25
                if target_pos[2] > self.table_offset[2] + 0.04:
                    reward += 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    # ──────────────────────────────────────────────────────────────────────────
    # Success check  (required by robomimic via EnvRobosuite.is_success)
    # ──────────────────────────────────────────────────────────────────────────

    def _check_success(self):
        """
        Task succeeds when the random target object is lifted above a target height.
        """
        target_pos = np.array(self.sim.data.body_xpos[self.lift_target_body_id])
        return target_pos[2] > self.table_offset[2] + 0.10