"""
GraspLiftEnv: Lightweight Gymnasium Environment for Training PPO Grasp & Lift Skills
===================================================================================
Wraps Robosuite's `TaskSorting` environment for fine-grained manipulation learning:
- Initial state: End-effector initialized in pre-grasp hover pose (12-18cm above object),
  matching the exact handoff state from TaskSpaceRRT.
- Shapes: Randomizes across both cubes (yellow_cube, purple_cube) and
  cylinders (red_can, blue_can, green_can) to train shape-adaptive grasping.
- Compact 12D observation space and 4D action space (dx, dy, dz, gripper)
  for fast, stable training with Stable-Baselines3 PPO.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import robosuite as suite
from src.envs.task_sorting import TaskSorting


# All pickable objects in the TaskSorting arena
ALL_TARGET_OBJECTS = [
    "yellow_cube",
    "purple_cube",
    "red_can",
    "blue_can",
    "green_can",
]


def extract_ppo_obs(
    sim_data: Any,
    eef_site_id: int,
    obj_body_id: int,
    gripper_qpos: np.ndarray,
    gripper_qvel: np.ndarray,
    table_z: float,
    initial_obj_z: float,
    is_grasped: bool,
    is_cylinder: bool,
) -> np.ndarray:
    """
    Extract a normalized 11D observation vector for the PPO policy.

    Components:
    - rel_pos (3): vector from EEF to object center (x, y, z)
    - gripper_qpos (2): left and right finger positions
    - gripper_qvel (2): left and right finger velocities
    - eef_z_rel (1): height of EEF above table surface
    - lift_height (1): current object z - initial object z
    - is_grasped (1): binary grasp indicator
    - is_cylinder (1): 0.0 for box/cube, 1.0 for cylinder/can
    """
    eef_pos = np.array(sim_data.site_xpos[eef_site_id])
    obj_pos = np.array(sim_data.body_xpos[obj_body_id])

    target_pos = obj_pos.copy()
    target_pos[2] += (0.035 if is_cylinder else 0.002)
    rel_pos = target_pos - eef_pos
    eef_z_rel = eef_pos[2] - table_z
    lift_height = obj_pos[2] - initial_obj_z

    obs_vec = np.concatenate([
        rel_pos,                                                  # [0:3]
        gripper_qpos,                                             # [3:5]
        gripper_qvel,                                             # [5:7]
        np.array([eef_z_rel], dtype=np.float32),                 # [7]
        np.array([lift_height], dtype=np.float32),               # [8]
        np.array([1.0 if is_grasped else 0.0], dtype=np.float32),# [9]
        np.array([1.0 if is_cylinder else 0.0], dtype=np.float32),# [10]
    ]).astype(np.float32)

    return obs_vec


class GraspLiftEnv(gym.Env):
    """
    Gymnasium-compliant environment for training PPO grasp and lift policies.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        target_name: Optional[str] = "yellow_cube",
        max_steps: int = 80,
        hover_height: float = 0.15,
        success_height: float = 0.06,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.target_name_override = target_name
        self.max_steps = max_steps
        self.hover_height = hover_height
        self.success_height = success_height
        self.render_mode = render_mode
        self._step_count = 0

        # Build headless Robosuite TaskSorting environment
        self.env = suite.make(
            env_name="TaskSorting",
            robots="Panda",
            has_renderer=(render_mode == "human"),
            has_offscreen_renderer=(render_mode == "rgb_array"),
            use_camera_obs=(render_mode == "rgb_array"),
            use_object_obs=True,
            control_freq=20,
            horizon=10000,
            ignore_done=True,
            hard_reset=False,
        )

        self.robot = self.env.robots[0]
        self.eef_site_id = self.robot.eef_site_id["right"]
        self.gripper = self.robot.gripper["right"] if isinstance(self.robot.gripper, dict) else self.robot.gripper
        self.gripper_joint_ids = [self.env.sim.model.joint_name2id(j) for j in self.gripper.joints]
        self.table_z = float(self.env.table_offset[2])

        # 4D Action Space: [dx, dy, dz, gripper] in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # 11D Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )

        self.current_target = "yellow_cube"
        self.target_body_id = 0
        self.initial_obj_z = self.table_z
        self.is_cylinder = False

    def _resolve_target(self, options: Optional[Dict[str, Any]] = None) -> str:
        if options and "target_name" in options:
            return options["target_name"]
        if self.target_name_override is not None:
            return self.target_name_override
        return "yellow_cube"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        # 1. Reset underlying simulation
        self.env.reset()

        # 2. Select active target object
        self.current_target = self._resolve_target(options)
        if hasattr(self.env, f"{self.current_target}_body_id"):
            self.target_body_id = getattr(self.env, f"{self.current_target}_body_id")
        else:
            for candidate in [self.current_target, f"{self.current_target}_main", f"{self.current_target}_root"]:
                try:
                    self.target_body_id = self.env.sim.model.body_name2id(candidate)
                    break
                except ValueError:
                    continue
        self.is_cylinder = "can" in self.current_target

        obj_pos = np.array(self.env.sim.data.body_xpos[self.target_body_id])
        self.initial_obj_z = float(obj_pos[2])

        # 3. Simulate RRT approach: step the robot open-loop to the pre-grasp hover pose
        # Target: directly above object with slight noise to model RRT arrival variance
        noise_xy = np.random.uniform(-0.015, 0.015, size=2)
        hover_target = obj_pos.copy()
        hover_target[:2] += noise_xy
        hover_target[2] += self.hover_height + np.random.uniform(0.0, 0.03)

        for _ in range(25):
            eef_pos = np.array(self.env.sim.data.site_xpos[self.eef_site_id])
            delta = hover_target - eef_pos
            if np.linalg.norm(delta) < 0.02:
                break
            action = np.zeros(7)
            action[:3] = np.clip(10.0 * delta, -1.0, 1.0)
            action[6] = -1.0  # open gripper
            self.env.step(action)

        # Update initial object z after physical settling
        self.initial_obj_z = float(self.env.sim.data.body_xpos[self.target_body_id][2])

        # 4. Extract policy observation
        obs = self._get_obs()
        info = {
            "target_name": self.current_target,
            "is_cylinder": self.is_cylinder,
            "initial_obj_z": self.initial_obj_z,
        }
        return obs, info

    def get_expert_action(self) -> np.ndarray:
        """
        Closed-loop expert demonstration policy for descend, grasp, and lift.
        Returns a 4D action [dx, dy, dz, gripper].
        """
        eef_pos = np.array(self.env.sim.data.site_xpos[self.eef_site_id])
        obj_pos = np.array(self.env.sim.data.body_xpos[self.target_body_id])
        target_obj = getattr(self.env, self.current_target)
        is_grasped = bool(self.env._check_grasp(gripper=self.gripper, object_geoms=target_obj))

        target_pos = obj_pos.copy()
        target_pos[2] += (0.035 if self.is_cylinder else 0.002)

        act = np.zeros(4, dtype=np.float32)
        delta = target_pos - eef_pos

        if delta[2] < -0.010:
            # EEF is above the grasp target: align and descend with gripper open
            act[:3] = np.clip(10.0 * delta, -1.0, 1.0)
            act[3] = -1.0
        elif not is_grasped:
            # EEF is at grasp height: align in XY and close gripper firmly
            act[:2] = np.clip(10.0 * delta[:2], -0.3, 0.3)
            act[2] = 0.0
            act[3] = 1.0
        else:
            # Grasped: lift vertically while holding gripper closed
            act[0] = 0.0
            act[1] = 0.0
            act[2] = 0.8
            act[3] = 1.0

        return act

    def _get_obs(self) -> np.ndarray:
        gripper_qpos = np.array([
            self.env.sim.data.qpos[j] for j in self.gripper_joint_ids
        ], dtype=np.float32)
        gripper_qvel = np.array([
            self.env.sim.data.qvel[j] for j in self.gripper_joint_ids
        ], dtype=np.float32)

        target_obj = getattr(self.env, self.current_target)
        is_grasped = bool(self.env._check_grasp(gripper=self.gripper, object_geoms=target_obj))

        return extract_ppo_obs(
            sim_data=self.env.sim.data,
            eef_site_id=self.eef_site_id,
            obj_body_id=self.target_body_id,
            gripper_qpos=gripper_qpos,
            gripper_qvel=gripper_qvel,
            table_z=self.table_z,
            initial_obj_z=self.initial_obj_z,
            is_grasped=is_grasped,
            is_cylinder=self.is_cylinder,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # Map 4D action [dx, dy, dz, gripper] to Robosuite 7D OSC action
        action = np.clip(action, -1.0, 1.0)
        full_action = np.zeros(7)
        full_action[:3] = action[:3]  # Translation deltas
        full_action[3:6] = 0.0        # Downward top-down orientation held steady
        # Firm binary gripper actuation (+1.0 close, -1.0 open)
        full_action[6] = 1.0 if action[3] > 0.0 else -1.0

        # Step simulator
        self.env.step(full_action)

        # Retrieve current state
        eef_pos = np.array(self.env.sim.data.site_xpos[self.eef_site_id])
        obj_pos = np.array(self.env.sim.data.body_xpos[self.target_body_id])
        target_obj = getattr(self.env, self.current_target)
        is_grasped = bool(self.env._check_grasp(gripper=self.gripper, object_geoms=target_obj))

        target_pos = obj_pos.copy()
        target_pos[2] += (0.035 if self.is_cylinder else 0.002)

        dist = float(np.linalg.norm(eef_pos - target_pos))
        lift_height = float(obj_pos[2] - self.initial_obj_z)

        # ── Shaped Reward Function ───────────────────────────────────────────
        # 1. Reaching reward: encourages descending smoothly to object center
        r_reach = 1.0 - np.tanh(10.0 * dist)

        # 2. Near-object gripper closure incentive
        if dist < 0.04:
            r_grip = 0.5 * (1.0 if action[3] > 0.0 else -0.5)
        elif action[3] > 0.2:
            r_grip = -0.2
        else:
            r_grip = 0.0

        # 3. Grasp reward: encourages closing fingers around object when aligned
        r_grasp = 2.0 if is_grasped else 0.0

        # 4. Lift reward: encourages upward vertical lifting once grasped
        r_lift = 20.0 * max(0.0, lift_height) if is_grasped else 0.0

        # 5. Small alive step cost to incentivize rapid task completion
        reward = float(r_reach + r_grip + r_grasp + r_lift - 0.01)

        # ── Termination Criteria ─────────────────────────────────────────────
        success = bool(lift_height >= self.success_height and is_grasped)
        terminated = False
        truncated = False

        if success:
            reward += 15.0  # Large sparse completion bonus
            terminated = True

        # Drop / fall off table failure condition
        if obj_pos[2] < self.table_z - 0.04:
            reward -= 2.0
            terminated = True

        # Episode horizon timeout
        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "is_success": success,
            "lift_height": lift_height,
            "is_grasped": is_grasped,
            "target_name": self.current_target,
            "dist": dist,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            frame = self.env.sim.render(height=512, width=512, camera_name="frontview")
            # Flip vertically to correct MuJoCo OpenGL inverted Y-axis without horizontal mirroring
            return np.ascontiguousarray(np.flipud(frame))
        return None

    def close(self):
        if self.env is not None:
            self.env.close()
