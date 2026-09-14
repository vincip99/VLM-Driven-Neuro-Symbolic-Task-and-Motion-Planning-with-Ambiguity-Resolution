"""
Action nodes (Skills) for Behavior Tree robotic execution.
Provides closed-loop manipulation primitives using TaskSpaceRRT motion planning.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Type

import numpy as np
import py_trees

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
# Pick-Up Skill (TaskSpaceRRT Motion Planning)
# ---------------------------------------------------------------------------
@register_skill("MotionPlanningPickUp")
@register_skill("PickUp")
@register_skill("Pick")
@register_skill("pick")
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
            if self.grasp_timer > 40:
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
# Place-In-Bin Skill (TaskSpaceRRT Motion Planning)
# ---------------------------------------------------------------------------
@register_skill("MotionPlanningPlaceInBin")
@register_skill("PlaceInBin")
@register_skill("Place")
@register_skill("place")
@register_skill("ThrowAway")
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