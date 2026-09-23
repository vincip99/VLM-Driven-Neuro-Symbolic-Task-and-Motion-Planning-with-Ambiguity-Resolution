"""
Action nodes (Skills) for Behavior Tree robotic execution.
Provides closed-loop manipulation primitives using TaskSpaceRRT motion planning
and Hybrid PPO Micro-Manipulation policies.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Type

# Eagerly import torch and PPO at module level to initialize PyTorch/CUDA runtime
# BEFORE any MuJoCo OpenGL/EGL offscreen rendering context is constructed.
try:
    import torch
    from stable_baselines3 import PPO
    _PPO_AVAILABLE = True
except ImportError:
    _PPO_AVAILABLE = False

import numpy as np
import py_trees

from src.envs import extract_ppo_obs
from .motion_planner import TaskSpaceRRT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------
SKILL_REGISTRY: Dict[str, Type["Skill"]] = {}


def register_skill(name: str):
    """Decorator to register a skill class by name."""
    def _wrap(cls: Type["Skill"]):
        SKILL_REGISTRY[name] = cls
        return cls
    return _wrap


# ---------------------------------------------------------------------------
# Base Skill
# ---------------------------------------------------------------------------
class Skill(py_trees.behaviour.Behaviour):
    """
    Base class for Behavior Tree action nodes.
    Each tick steps the simulation environment until is_done() returns True.
    """

    def __init__(self, name: str, args: Dict[str, Any], env=None, max_steps: int = 400):
        super().__init__(name=name)
        self.args = dict(args) if args else {}
        self.env = env
        self.max_steps = max_steps
        self._steps = 0

    def setup(self, **kwargs):
        """Called once before the tree ticks. Binds the environment."""
        self.env = kwargs.get("env", self.env)

    def initialise(self):
        """Called when this skill becomes active (RUNNING)."""
        self._steps = 0
        self.logger.debug(f"[{self.name}] Initialized with args={self.args}")

    def update(self) -> py_trees.common.Status:
        """Main execution tick: inspect observation, compute action, step simulator."""
        if self.env is None:
            self.logger.error(f"[{self.name}] No environment bound.")
            return py_trees.common.Status.FAILURE

        obs = self.env.get_obs()

        if self.is_done(obs):
            self.logger.info(f"[{self.name}] SUCCESS after {self._steps} steps")
            return py_trees.common.Status.SUCCESS

        if self._steps >= self.max_steps:
            self.logger.warning(f"[{self.name}] FAILURE - timeout at {self._steps} steps")
            return py_trees.common.Status.FAILURE

        action = self._get_action(obs)
        self.env.step(action)
        self._steps += 1
        return py_trees.common.Status.RUNNING

    def is_done(self, obs: dict) -> bool:
        """Subclasses must implement the completion condition."""
        raise NotImplementedError(f"{self.name}: is_done() not implemented")

    def _get_action(self, obs: dict) -> np.ndarray:
        """Subclasses must return a 7-DoF robot action vector."""
        raise NotImplementedError(f"{self.name}: _get_action() not implemented")


# ---------------------------------------------------------------------------
# Pick-Up Skill (Heuristic TaskSpaceRRT Motion Planning)
# ---------------------------------------------------------------------------
@register_skill("heuristic_pick")
class MotionPlanningPickUpSkill(Skill):
    """
    Picks up an object using collision-free TaskSpaceRRT navigation
    followed by a staged descend-grasp-lift state machine.
    """

    def initialise(self):
        super().initialise()
        self.stage = "follow_path"
        self.hover_height = 0.20
        self.kp = 10.0
        self.max_speed = 1.0
        self.grasp_timer = 0
        self.path = []
        self.current_wp_idx = 0
        self._wp_timer = 0
        self._descend_timer = 0
        self._lift_timer = 0

        target_obj = self.args.get("object", self.args.get("ob", ""))
        self.descend_height = 0.002 if "cube" in target_obj else 0.035

        if self.env is None:
            return
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is None:
            return

        obj_pos = self.env._base_env.sim.data.body_xpos[body_id]
        self._initial_obj_z = obj_pos[2]

        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]

        # Extract table obstacles (bin and pot) to avoid during transit
        obstacles = []
        bin_pos = self.env._resolve_bin_pos("bin")
        if bin_pos is not None and np.linalg.norm(obj_pos[:2] - bin_pos[:2]) > 0.18:
            obstacles.append({
                "min": bin_pos - np.array([0.16, 0.16, 0.05]),
                "max": bin_pos + np.array([0.16, 0.16, 0.20])
            })
        pot_pos = self.env._resolve_bin_pos("pot")
        if pot_pos is not None and np.linalg.norm(obj_pos[:2] - pot_pos[:2]) > 0.18:
            obstacles.append({
                "min": pot_pos - np.array([0.15, 0.18, 0.05]),
                "max": pot_pos + np.array([0.15, 0.18, 0.20])
            })

        # Plan RRT trajectory to hover position directly above object
        planner = TaskSpaceRRT()
        goal_hover = obj_pos.copy()
        goal_hover[2] += self.hover_height
        self.path = planner.plan(eef_pos, goal_hover, obstacles)
        self.logger.info(f"[{self.name}] Generated path with {len(self.path)} waypoints")

    def is_done(self, obs: dict) -> bool:
        if self.stage != "lift":
            return False
        target_obj = self.args.get("object", self.args.get("ob", ""))
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is not None:
            obj_z = self.env._base_env.sim.data.body_xpos[body_id][2]
            init_z = getattr(self, "_initial_obj_z", self.env._table_height)
            if obj_z > init_z + 0.02:
                return True
        return self.env.is_grasped(target_obj)

    def _get_action(self, obs: dict) -> np.ndarray:
        target_obj = self.args.get("object", self.args.get("ob", ""))
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is None:
            return np.zeros(7)

        obj_pos = self.env._base_env.sim.data.body_xpos[body_id]
        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]

        action = np.zeros(7)
        action[6] = -1.0  # Open gripper by default

        # ── Stage 1: Traverse RRT Waypoints to Hover Pose ──────────
        if self.stage == "follow_path":
            self._wp_timer += 1
            if self.current_wp_idx < len(self.path):
                target_pos = self.path[self.current_wp_idx]
                if np.linalg.norm(eef_pos - target_pos) < 0.04 or self._wp_timer > 30:
                    self.current_wp_idx += 1
                    self._wp_timer = 0
            else:
                self.stage = "descend"
                target_pos = eef_pos

        # ── Stage 2: Descend to Grasp Height ───────────────────────
        if self.stage == "descend":
            target_pos = obj_pos.copy()
            target_pos[2] += self.descend_height
            self._descend_timer += 1

            xy_err = np.linalg.norm(eef_pos[:2] - target_pos[:2])
            z_err = abs(eef_pos[2] - target_pos[2])
            z_tol = 0.015 if "cube" in target_obj else 0.025
            if (xy_err < 0.025 and z_err < z_tol) or self._descend_timer > 60:
                self.stage = "grasp"
                self.grasp_timer = 0

        # ── Stage 3: Close Gripper ─────────────────────────────────
        elif self.stage == "grasp":
            action[6] = 1.0  # Close gripper firmly
            self.grasp_timer += 1
            target_pos = obj_pos.copy()
            target_pos[2] += self.descend_height
            if self.grasp_timer > 20:
                self.stage = "lift"
                self._lift_timer = 0
                self._initial_obj_z = obj_pos[2]

        # ── Stage 4: Lift Object ───────────────────────────────────
        elif self.stage == "lift":
            target_pos = obj_pos.copy()
            target_pos[2] = self._initial_obj_z + self.hover_height
            action[6] = 1.0  # Keep closed
            self._lift_timer += 1
            # Retry if grasp missed (object did not lift)
            if self._lift_timer > 40 and (obj_pos[2] < self._initial_obj_z + 0.015):
                self.stage = "descend"
                self._descend_timer = 0
                action[6] = -1.0

        # Compute proportional control delta
        delta = target_pos - eef_pos
        if self.stage == "follow_path":
            action[:3] = np.clip(self.kp * delta, -self.max_speed, self.max_speed)
        elif self.stage == "descend":
            action[:3] = np.clip(self.kp * delta, -0.60, 0.60)
        elif self.stage == "grasp":
            action[:3] = np.clip(self.kp * delta, -0.15, 0.15)
        elif self.stage == "lift":
            action[:3] = np.clip(self.kp * delta, -0.35, 0.35)

        return action


# ---------------------------------------------------------------------------
# Hybrid Pick-Up Skill (TaskSpaceRRT Macro-Navigation + PPO Micro-Manipulation)
# ---------------------------------------------------------------------------
@register_skill("pick")
@register_skill("hybrid_ppo_pick")
class HybridPPOPickUpSkill(MotionPlanningPickUpSkill):
    """
    Hybrid Pick-Up Skill combining Macro-Navigation via TaskSpaceRRT and
    Micro-Manipulation via a trained PPO Reinforcement Learning Policy.

    Execution Lifecycle:
    1. 'follow_path': TaskSpaceRRT navigates collision-free from arbitrary start
       pose to a pre-grasp hover pose (~18cm directly above target object).
    2. 'ppo_lift': Once at the hover pose, the trained PPO policy takes over,
       processing the 11D observation vector (relative EEF-object translation,
       finger states, cylinder vs cube flag) and issuing 4D compliant commands
       (dx, dy, dz, gripper) to descend, grasp, and lift the object.
    3. Automatic Heuristic Fallback: If PPO model checkpoint is not found, or if
       PPO is disabled via env/args, or if PPO exceeds its step budget without
       lifting, the skill seamlessly falls back to the deterministic staged
       controller (descend -> grasp -> lift).
    """

    _cached_model = None
    _cached_model_path: Optional[str] = None

    def __init__(self, name: str, args: Dict[str, Any], env=None, max_steps: int = 400):
        super().__init__(name=name, args=args, env=env, max_steps=max_steps)
        self.use_ppo = bool(self.args.get("use_ppo", True))
        self.model_path = self.args.get("model_path", "models/ppo_hybrid_lift.zip")
        self._ppo_steps = 0
        self._max_ppo_steps = 90

    @classmethod
    def get_model(cls, model_path: str):
        if cls._cached_model is not None and cls._cached_model_path == model_path:
            return cls._cached_model

        candidates = [
            model_path,
            os.path.join(os.path.dirname(model_path), "best_model", "best_model.zip"),
            os.path.abspath(model_path),
        ]
        found_path = None
        for c in candidates:
            if c and os.path.isfile(c):
                found_path = c
                break

        if found_path is None:
            return None

        if not _PPO_AVAILABLE:
            log.warning("stable_baselines3 is not available in environment.")
            return None

        try:
            cls._cached_model = PPO.load(found_path, device="cpu")
            cls._cached_model_path = model_path
            log.info(f"Loaded PPO checkpoint from {found_path}")
            return cls._cached_model
        except Exception as e:
            log.warning(f"Failed loading PPO model from {found_path}: {e}")
            return None

    def initialise(self):
        super().initialise()
        self._ppo_steps = 0
        self._ppo_lifting = False
        self._ppo_closing_timer = 0
        # Check if env has a global use_ppo flag or model path override
        if self.env is not None and hasattr(self.env, "use_ppo"):
            self.use_ppo = bool(self.env.use_ppo)
        if self.env is not None and hasattr(self.env, "ppo_model_path") and self.env.ppo_model_path:
            self.model_path = str(self.env.ppo_model_path)

    def _check_contact_grasp(self, target_obj: str) -> bool:
        """Return True if gripper fingers are physically in contact with the object."""
        if self.env is None or not hasattr(self.env, "_base_env"):
            return False
        base = self.env._base_env
        gripper = base.robots[0].gripper["right"]
        target_geom = getattr(base, target_obj, None)
        if target_geom is None:
            return False
        if base._check_grasp(gripper=gripper, object_geoms=target_geom):
            return True
        left_pad = gripper.important_geoms.get("left_fingerpad", [])
        right_pad = gripper.important_geoms.get("right_fingerpad", [])
        c_left = base.check_contact(left_pad, target_geom.contact_geoms)
        c_right = base.check_contact(right_pad, target_geom.contact_geoms)
        return bool(c_left and c_right)

    def is_done(self, obs: dict) -> bool:
        target_obj = self.args.get("object", self.args.get("ob", ""))
        body_id = self.env._resolve_body_id(target_obj) if self.env else None

        if body_id is not None:
            obj_z = self.env._base_env.sim.data.body_xpos[body_id][2]
            init_z = getattr(self, "_initial_obj_z", self.env._table_height)
            is_grasped = self._check_contact_grasp(target_obj) or self.env.is_grasped(target_obj)
            # Successfully lifted if higher than initial position and grasped
            if obj_z > init_z + 0.025 and is_grasped:
                return True
            # Unconditional success if lifted significantly
            if obj_z > init_z + 0.045:
                return True

        if self.stage in ["lift", "descend", "grasp"]:
            return super().is_done(obs)

        return False

    def _get_action(self, obs: dict) -> np.ndarray:
        target_obj = self.args.get("object", self.args.get("ob", ""))
        body_id = self.env._resolve_body_id(target_obj) if self.env else None
        if body_id is None:
            return np.zeros(7)

        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
        obj_pos = self.env._base_env.sim.data.body_xpos[body_id]

        # ── Stage 1: Traverse RRT Waypoints to Pre-Grasp Hover Pose ─
        if self.stage == "follow_path":
            self._wp_timer += 1
            if self.current_wp_idx < len(self.path) - 1:
                target_pos = self.path[self.current_wp_idx]
                if np.linalg.norm(eef_pos - target_pos) < 0.04 or self._wp_timer > 30:
                    self.current_wp_idx += 1
                    self._wp_timer = 0
            elif self.current_wp_idx == len(self.path) - 1:
                target_pos = self.path[-1]
                xy_err = np.linalg.norm(eef_pos[:2] - target_pos[:2])
                if xy_err < 0.015 or self._wp_timer > 35:
                    self.current_wp_idx += 1
                    self._wp_timer = 0
            else:
                # Reached pre-grasp hover pose: use PPO for cubes
                is_cube = "cube" in target_obj
                ppo_model = self.get_model(self.model_path) if (self.use_ppo and is_cube) else None
                if ppo_model is not None:
                    self.stage = "ppo_lift"
                    self._ppo_steps = 0
                    self._ppo_lifting = False
                    self._ppo_closing_timer = 0
                    self._initial_obj_z = obj_pos[2]
                    self.logger.info(f"[{self.name}] Pre-grasp hover reached. Handing over to PPO micro-policy.")
                else:
                    self.stage = "descend"
                    self.logger.info(f"[{self.name}] PPO unavailable, disabled, or target is cylinder. Using heuristic descend-grasp-lift.")
                target_pos = eef_pos

            delta = target_pos - eef_pos
            action = np.zeros(7)
            action[:3] = np.clip(self.kp * delta, -self.max_speed, self.max_speed)
            action[6] = -1.0  # Open gripper
            return action

        # ── Stage 2: PPO Micro-Manipulation (Compliant Descent, Grasp, Lift) ─
        elif self.stage == "ppo_lift":
            self._ppo_steps += 1
            # Check timeout / fallback to deterministic controller
            if self._ppo_steps > self._max_ppo_steps:
                self.logger.warning(
                    f"[{self.name}] PPO step budget ({self._max_ppo_steps}) reached. Falling back to heuristic controller."
                )
                self.stage = "descend"
                self._descend_timer = 0
                return super()._get_action(obs)

            init_z = getattr(self, "_initial_obj_z", self.env._table_height)
            table_z = getattr(self.env, "_table_height", 0.8)

            # Once lift phase is triggered, smoothly lift to hover height
            if getattr(self, "_ppo_lifting", False):
                action = np.zeros(7)
                target_lift_z = init_z + self.hover_height
                z_delta = target_lift_z - eef_pos[2]
                action[2] = np.clip(self.kp * z_delta, 0.15, 0.60)
                xy_delta = obj_pos[:2] - eef_pos[:2]
                action[:2] = np.clip(self.kp * xy_delta, -0.20, 0.20)
                action[6] = 1.0  # Keep gripper firmly closed
                return action

            # Extract 11D observation vector
            sim = self.env._base_env.sim
            robot = self.env._base_env.robots[0]
            gripper = robot.gripper["right"]
            gripper_joint_ids = [sim.model.joint_name2id(j) for j in gripper.joints]
            gripper_qpos = np.array([sim.data.qpos[j] for j in gripper_joint_ids], dtype=np.float32)
            gripper_qvel = np.array([sim.data.qvel[j] for j in gripper_joint_ids], dtype=np.float32)
            is_contact = self._check_contact_grasp(target_obj)
            is_grasped = is_contact or bool(self.env.is_grasped(target_obj))

            ppo_obs = extract_ppo_obs(
                sim_data=sim.data,
                eef_site_id=eef_site_id,
                obj_body_id=body_id,
                gripper_qpos=gripper_qpos,
                gripper_qvel=gripper_qvel,
                table_z=table_z,
                initial_obj_z=self._initial_obj_z,
                is_grasped=is_grasped,
                is_cylinder=False,
            )

            model = self.get_model(self.model_path)
            if model is None:
                self.stage = "descend"
                return super()._get_action(obs)

            action_4d, _ = model.predict(ppo_obs, deterministic=True)
            action = np.zeros(7)
            action[:3] = np.clip(action_4d[:3], -1.0, 1.0)
            action[3:6] = 0.0
            action[6] = 1.0 if action_4d[3] > 0.0 else -1.0

            # Prevent EEF from driving into table and jamming gripper fingers
            safe_min_z = table_z + 0.016
            if eef_pos[2] <= safe_min_z and action[2] < 0:
                action[2] = 0.0

            # Grasp closure phase: when PPO closes gripper or EEF reaches grasp height
            eef_obj_z_diff = eef_pos[2] - obj_pos[2]
            is_at_grasp_height = eef_obj_z_diff < 0.018
            if action[6] > 0.0 or is_at_grasp_height:
                action[6] = 1.0  # Firm binary closure
                self._ppo_closing_timer += 1
                xy_delta = obj_pos[:2] - eef_pos[:2]
                action[:2] = np.clip(5.0 * xy_delta, -0.15, 0.15)
                target_z = obj_pos[2] + 0.003
                action[2] = np.clip(5.0 * (target_z - eef_pos[2]), -0.10, 0.10)

                # Trigger lift once grasped or after fingers close
                if is_contact or self._ppo_closing_timer > 15:
                    self._ppo_lifting = True
                    action[2] = 0.5
                    action[6] = 1.0

            return action

        # ── Fallback Stages: descend, grasp, lift via superclass controller ─
        else:
            return super()._get_action(obs)


# ---------------------------------------------------------------------------
# Place-In-Bin Skill (TaskSpaceRRT Motion Planning)
# ---------------------------------------------------------------------------
@register_skill("place")
class MotionPlanningPlaceInBinSkill(Skill):
    """
    Transports held object to a target container (bin or pot) using TaskSpaceRRT,
    then opens the gripper to release it.
    """

    def initialise(self):
        super().initialise()
        self.stage = "follow_path"
        self.hover_height = 0.18
        self.kp = 10.0
        self.max_speed = 1.0
        self.release_timer = 0
        self.path = []
        self.current_wp_idx = 0
        self._wp_timer = 0

        if self.env is None:
            return

        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]

        target_asset = self.args.get("asset", self.args.get("pos", "bin"))
        target_container_pos = self.env._resolve_bin_pos(target_asset)
        if target_container_pos is None:
            self.logger.warning(f"[{self.name}] Container '{target_asset}' not found!")
            return

        # Avoid the other container during transit
        obstacles = []
        asset_str = str(target_asset).lower().strip()
        if asset_str in ["pot", "bowl"]:
            other_pos = self.env._resolve_bin_pos("bin")
            if other_pos is not None:
                obstacles.append({
                    "min": other_pos - np.array([0.16, 0.16, 0.05]),
                    "max": other_pos + np.array([0.16, 0.16, 0.20])
                })
        else:
            other_pos = self.env._resolve_bin_pos("pot")
            if other_pos is not None:
                obstacles.append({
                    "min": other_pos - np.array([0.15, 0.18, 0.05]),
                    "max": other_pos + np.array([0.15, 0.18, 0.20])
                })

        # Plan collision-free path to hover above destination
        planner = TaskSpaceRRT()
        self.goal_hover = target_container_pos.copy()
        self.goal_hover[2] += self.hover_height
        self.path = planner.plan(eef_pos, self.goal_hover, obstacles)
        self.logger.info(f"[{self.name}] Generated path with {len(self.path)} waypoints")

    def is_done(self, obs: dict) -> bool:
        if self.stage != "release":
            return False
        if self.release_timer < 20:
            return False
        return self.env.is_in_bin(
            self.args.get("object", self.args.get("ob", "")),
            self.args.get("asset", self.args.get("pos", "bin")),
        )

    def _get_action(self, obs: dict) -> np.ndarray:
        if self.env is None:
            return np.zeros(7)

        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]

        action = np.zeros(7)
        action[6] = 1.0  # Keep gripper closed during transit
        target_pos = eef_pos

        # ── Stage 1: Traverse RRT Waypoints to Container ───────────
        if self.stage == "follow_path":
            self._wp_timer += 1
            if self.current_wp_idx < len(self.path):
                target_pos = self.path[self.current_wp_idx]
                dist = np.linalg.norm(eef_pos - target_pos)
                is_final_wp = (self.current_wp_idx == len(self.path) - 1)
                timeout = 80 if is_final_wp else 40
                threshold = 0.04 if is_final_wp else 0.06
                if dist < threshold or self._wp_timer > timeout:
                    self.current_wp_idx += 1
                    self._wp_timer = 0
            else:
                self.stage = "release"

        # ── Stage 2: Open Gripper and Release Object ───────────────
        elif self.stage == "release":
            action[6] = -1.0  # Open gripper
            self.release_timer += 1
            if hasattr(self, "goal_hover"):
                target_pos = self.goal_hover

        delta = target_pos - eef_pos
        action[:3] = np.clip(self.kp * delta, -self.max_speed, self.max_speed)
        return action