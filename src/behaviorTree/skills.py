"""
Leaf "Action" nodes of the behavior tree, wired to a real robomimic BC policy.

Each Skill subclass is registered in SKILL_REGISTRY so the BT builder can
instantiate it by name from a VLM-generated JSON plan.

Policy loading
--------------
Both PickUp and PlaceInBin share the SAME checkpoint (the BC policy learned
the full pick-and-place trajectory). The BT decides *which* object to pick
and *where* to place it; `is_done()` determines when each phase completes:
  - PickUp   → done when gripper is closed + object is lifted above table
  - PlaceInBin → done when object's XY is within the bin footprint

The policy instance is loaded once per skill node in `setup()` and is
stateless (MLP, no recurrence), so sharing or re-loading is fine.
"""
from __future__ import annotations
import os
import glob
import logging
import numpy as np
import py_trees
from typing import Any, Dict, Optional

from .motion_planner import TaskSpaceRRT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SKILL_REGISTRY: Dict[str, Type["Skill"]] = {}


def register_skill(name: str):
    def _wrap(cls: Type["Skill"]):
        SKILL_REGISTRY[name] = cls
        return cls
    return _wrap


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
_BC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "bc_can", "bc_mlp_example")


def _find_latest_ckpt(bc_dir: str) -> Optional[str]:
    """Return the most-recently-written .pth file under bc_dir."""
    bc_dir = os.path.abspath(bc_dir)
    # prefer last.pth (most recent epoch)
    candidates = glob.glob(os.path.join(bc_dir, "*", "last.pth"))
    if not candidates:
        candidates = glob.glob(os.path.join(bc_dir, "*", "models", "*.pth"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


CKPT_PATH: Optional[str] = _find_latest_ckpt(_BC_DIR)
if CKPT_PATH:
    log.info(f"[skills] auto-discovered checkpoint: {CKPT_PATH}")
else:
    log.warning(f"[skills] no checkpoint found under {_BC_DIR}")

# Singleton cache: load the policy once, share across all skill nodes
_POLICY_CACHE: Optional[Any] = None


def _get_shared_policy():
    """Load the policy from CKPT_PATH once and cache it for reuse."""
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    if CKPT_PATH is None:
        return None
    try:
        from robomimic.utils.file_utils import policy_from_checkpoint
        policy, _ = policy_from_checkpoint(ckpt_path=CKPT_PATH)
        log.info(f"[skills] policy loaded and cached from {os.path.basename(CKPT_PATH)}")
        _POLICY_CACHE = policy
        return policy
    except Exception as exc:
        log.error(f"[skills] policy load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Base Skill
# ---------------------------------------------------------------------------
class Skill(py_trees.behaviour.Behaviour):
    """
    Base class for a Generic Skill.

    Subclasses must implement:
      - is_done(obs) → bool  : termination condition for this phase

    Optionally override:
      - load_policy()         : return a different policy (e.g. per-object ckpt)
    """

    def __init__(self, name: str, args: Dict[str, Any], env=None, max_steps: int = 400):
        super().__init__(name=name)
        self.args = args
        self.env = env          # injected by runner or tree.setup()
        self.policy = None
        self.max_steps = max_steps
        self._steps = 0

    # ── py_trees lifecycle ─────────────────────────────────────────────────

    def setup(self, **kwargs):
        """Called once before the tree starts ticking. Load env + policy."""
        self.env = kwargs.get("env", self.env)
        self.policy = self.load_policy()

    def initialise(self):
        """Called each time the skill becomes active (enters RUNNING)."""
        self._steps = 0
        # reset policy hidden state (no-op for MLP, matters for LSTM)
        if self.policy is not None and hasattr(self.policy, "start_episode"):
            self.policy.start_episode()
        self.logger.debug(f"[{self.name}] start — args={self.args}")

    def update(self) -> py_trees.common.Status:
        if self.env is None:
            self.logger.error(f"[{self.name}] no env bound — call setup(env=...)")
            return py_trees.common.Status.FAILURE

        obs = self.env.get_obs()

        if self.is_done(obs):
            self.logger.info(f"[{self.name}] SUCCESS after {self._steps} steps")
            return py_trees.common.Status.SUCCESS

        if self._steps >= self.max_steps:
            self.logger.warning(f"[{self.name}] FAILURE — timeout at {self._steps} steps")
            return py_trees.common.Status.FAILURE

        action = self._get_action(obs)
        self.env.step(action)
        self._steps += 1
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status):
        self.logger.debug(f"[{self.name}] terminated → {new_status} ({self._steps} steps)")

    # ── to override ────────────────────────────────────────────────────────

    def load_policy(self):
        """
        Return the shared cached robomimic policy (loaded once from CKPT_PATH).
        Override to use a per-object or per-skill checkpoint instead.
        """
        return _get_shared_policy()

    def is_done(self, obs: dict) -> bool:
        raise NotImplementedError(f"{self.name}: implement is_done()")

    def _get_action(self, obs: dict) -> np.ndarray:
        if self.policy is not None:
            # Re-map the object observation to match PickPlaceCan policy expectations.
            # The PickPlaceCan policy expects a 14-dim object observation:
            # [rel_pos (3), rel_quat (4), obj_pos (3), obj_quat (4)]
            if self.env is not None:
                target_obj = self.args.get("object", "")
                body_id = self.env._resolve_body_id(target_obj)
                
                if body_id is not None:
                    # Get object pose
                    obj_pos = self.env._base_env.sim.data.body_xpos[body_id]
                    obj_quat = self.env._base_env.sim.data.body_xquat[body_id]
                    
                    from robosuite.utils import transform_utils as T
                    obj_quat_xyzw = T.convert_quat(obj_quat, to="xyzw")
                    
                    # Get EEF pose
                    eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
                    eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
                    eef_mat = self.env._base_env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
                    eef_quat_xyzw = T.mat2quat(eef_mat)
                    
                    # Compute relative position and quaternion (object pose in EEF frame)
                    # This matches robosuite's _get_rel_obj_eef_sensor exactly
                    obj_pose = T.pose2mat((obj_pos, obj_quat_xyzw))
                    world_pose_in_gripper = T.pose_inv(T.pose2mat((eef_pos, eef_quat_xyzw)))
                    rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, world_pose_in_gripper)
                    rel_pos, rel_quat = T.mat2pose(rel_pose)
                    
                    custom_obj_obs = np.zeros(14)
                    custom_obj_obs[0:3] = rel_pos
                    custom_obj_obs[3:7] = rel_quat
                    custom_obj_obs[7:10] = obj_pos
                    custom_obj_obs[10:14] = obj_quat_xyzw
                    
                    obs["object"] = custom_obj_obs

            return self.policy(ob=obs)
        raise RuntimeError(f"{self.name}: no policy and no _get_action override")


# ---------------------------------------------------------------------------
# Concrete skills
# ---------------------------------------------------------------------------

@register_skill("PickUp")
class PickUpSkill(Skill):
    """
    Run the BC policy until the target object is grasped and lifted.
    """

    def is_done(self, obs: dict) -> bool:
        if self.policy is None and hasattr(self, "_mp_skill"):
            return self._mp_skill.is_done(obs)
        return self.env.is_grasped(self.args.get("object", ""))

    def _get_action(self, obs: dict) -> np.ndarray:
        if self.policy is not None:
            return super()._get_action(obs)
        if not hasattr(self, "_mp_skill"):
            self._mp_skill = MotionPlanningPickUpSkill(name=self.name, args=self.args, env=self.env)
            self._mp_skill.initialise()
        return self._mp_skill._get_action(obs)


@register_skill("PlaceInBin")
class PlaceInBinSkill(Skill):
    """
    Continue running the BC policy until the object is inside the bin.
    """

    def is_done(self, obs: dict) -> bool:
        if self.policy is None and hasattr(self, "_mp_skill"):
            return self._mp_skill.is_done(obs)
        return self.env.is_in_bin(
            self.args.get("object", ""),
            self.args.get("asset", "bin"),
        )

    def _get_action(self, obs: dict) -> np.ndarray:
        if self.policy is not None:
            return super()._get_action(obs)
        if not hasattr(self, "_mp_skill"):
            self._mp_skill = MotionPlanningPlaceInBinSkill(name=self.name, args=self.args, env=self.env)
            self._mp_skill.initialise()
        return self._mp_skill._get_action(obs)


@register_skill("ThrowAway")
class ThrowAwaySkill(PlaceInBinSkill):
    """
    Discard the held object. Same mechanics as PlaceInBin; kept as a separate
    vocabulary token so the VLM can distinguish intent (sorting vs discarding).
    """
    pass

@register_skill("MotionPlanningPickUp")
class MotionPlanningPickUpSkill(Skill):
    """
    Fallback motion planning controller to pick up an object.
    Uses TaskSpaceRRT to avoid the bin, then descends to grasp.
    """

    def initialise(self):
        super().initialise()
        self.stage = "follow_path"
        self.hover_height = 0.20
        target_obj = self.args.get("object", "")
        if "cube" in target_obj:
            self.descend_height = 0.002  # EEF site at 0.002m above cube center places finger pads directly on cube
        else:
            self.descend_height = 0.035  # EEF site at 0.035m above object center places finger pads around cylinder body
        self.kp = 10.0
        self.max_speed = 1.0
        self.grasp_timer = 0
        
        self.path = []
        self.current_wp_idx = 0
        
        # 1. Get targets
        target_obj = self.args.get("object", "")
        if self.env is None: return
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is None: return
        
        obj_pos = self.env._base_env.sim.data.body_xpos[body_id]
        
        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
        
        # 2. Extract Obstacles (Bin and Pot)
        obstacles = []
        bin_pos = self.env._resolve_bin_pos("bin")
        if bin_pos is not None and np.linalg.norm(obj_pos[:2] - bin_pos[:2]) > 0.18:
            obstacles.append({
                'min': bin_pos - np.array([0.16, 0.16, 0.05]),
                'max': bin_pos + np.array([0.16, 0.16, 0.20])
            })
        pot_pos = self.env._resolve_bin_pos("pot")
        if pot_pos is not None and np.linalg.norm(obj_pos[:2] - pot_pos[:2]) > 0.18:
            obstacles.append({
                'min': pot_pos - np.array([0.15, 0.18, 0.05]),
                'max': pot_pos + np.array([0.15, 0.18, 0.20])
            })
            
        # 3. Plan Path
        planner = TaskSpaceRRT()
        goal_hover = obj_pos.copy()
        goal_hover[2] += self.hover_height
        self.path = planner.plan(eef_pos, goal_hover, obstacles)
        self.logger.info(f"[{self.name}] Generated path with {len(self.path)} waypoints")

    def load_policy(self):
        return None

    def is_done(self, obs: dict) -> bool:
        if self.stage != "lift":
            return False
        target_obj = self.args.get("object", "")
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is not None:
            obj_z = self.env._base_env.sim.data.body_xpos[body_id][2]
            init_z = getattr(self, "_initial_obj_z", self.env._table_height)
            if obj_z > init_z + 0.02:
                return True
        return self.env.is_grasped(target_obj)

    def _get_action(self, obs: dict) -> np.ndarray:
        target_obj = self.args.get("object", "")
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is None:
            return np.zeros(7)

        obj_pos = self.env._base_env.sim.data.body_xpos[body_id]
        
        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
        
        action = np.zeros(7)
        action[6] = -1.0 # open gripper
        
        if self.stage == "follow_path":
            if not hasattr(self, "_wp_timer"):
                self._wp_timer = 0
            self._wp_timer += 1
            if self.current_wp_idx < len(self.path):
                target_pos = self.path[self.current_wp_idx]
                if np.linalg.norm(eef_pos - target_pos) < 0.04 or self._wp_timer > 30:
                    self.current_wp_idx += 1
                    self._wp_timer = 0
            else:
                self.stage = "descend"
                target_pos = eef_pos # hold
        
        if self.stage == "descend":
            target_pos = obj_pos.copy()
            target_pos[2] += self.descend_height
            if not hasattr(self, "_descend_timer"):
                self._descend_timer = 0
            self._descend_timer += 1
            # Transition to grasp when fully descended to target height
            xy_err = np.linalg.norm(eef_pos[:2] - target_pos[:2])
            z_err = abs(eef_pos[2] - target_pos[2])
            z_tol = 0.015 if "cube" in target_obj else 0.025
            if (xy_err < 0.025 and z_err < z_tol) or self._descend_timer > 60:
                self.stage = "grasp"
                self.grasp_timer = 0
                
        elif self.stage == "grasp":
            action[6] = 1.0 # close gripper firmly
            self.grasp_timer += 1
            target_pos = obj_pos.copy()
            target_pos[2] += self.descend_height
            if self.grasp_timer > 40:
                self.stage = "lift"
                self._lift_timer = 0
                self._initial_obj_z = obj_pos[2]
                
        elif self.stage == "lift":
            target_pos = obj_pos.copy()
            target_pos[2] = self._initial_obj_z + self.hover_height
            action[6] = 1.0 # keep closed
            if not hasattr(self, "_lift_timer"):
                self._lift_timer = 0
            self._lift_timer += 1
            # Retry if grasp missed (object failed to lift)
            if self._lift_timer > 40 and (obj_pos[2] < self._initial_obj_z + 0.015):
                print(f"[{self.name}] Grasp missed (obj_z={obj_pos[2]:.2f} vs init={self._initial_obj_z:.2f}), retrying...")
                self.stage = "descend"
                self._descend_timer = 0
                action[6] = -1.0 # re-open gripper
        
        if self.stage == "follow_path":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -self.max_speed, self.max_speed)
        elif self.stage == "descend":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -0.60, 0.60)
        elif self.stage == "grasp":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -0.15, 0.15)  # Hold steady while closing
        elif self.stage == "lift":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -0.35, 0.35)  # Upward lift
            
        print(f"[{self.name}] stage={self.stage}, eef_z={eef_pos[2]:.2f}, obj_z={obj_pos[2]:.2f}")
        return action


@register_skill("HybridPPOPickUp")
@register_skill("PPOLift")
class HybridPPOPickUpSkill(Skill):
    """
    Hybrid Skill: Uses TaskSpaceRRT for global navigation to pre-grasp position,
    then hands control over to a PPO policy for the grasp and lift phase.
    
    If the PPO checkpoint is not yet loaded, it seamlessly uses the calibrated
    heuristic controller as a fallback.
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

        target_obj = self.args.get("object", "")
        if "cube" in target_obj:
            self.descend_height = 0.015
        else:
            self.descend_height = 0.035
        if self.env is None: return
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is None: return
        
        obj_pos = self.env._base_env.sim.data.body_xpos[body_id]
        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
        
        obstacles = []
        bin_pos = self.env._resolve_bin_pos("bin")
        if bin_pos is not None:
            obstacles.append({
                'min': bin_pos - np.array([0.15, 0.15, 0.05]),
                'max': bin_pos + np.array([0.15, 0.15, 0.20])
            })
            
        planner = TaskSpaceRRT()
        goal_hover = obj_pos.copy()
        goal_hover[2] += self.hover_height
        self.path = planner.plan(eef_pos, goal_hover, obstacles)
        self.logger.info(f"[{self.name}] Initialized RRT path with {len(self.path)} waypoints")

    def load_policy(self):
        """
        Loads user PPO checkpoint from args['ckpt_path'] or default locations.
        Supports PyTorch (.pt, .pth) and Stable-Baselines3 (.zip).
        """
        custom_path = self.args.get("ckpt_path")
        search_paths = []
        if custom_path:
            search_paths.append(custom_path)
            
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        search_paths.extend([
            os.path.join(project_root, "saved_agents", "ppo_lift", "lift.zip"),
            os.path.join(project_root, "saved_agents", "ppo_lift", "lift.pth"),
            os.path.join(project_root, "saved_agents", "ppo_lift", "lift.pt"),
            os.path.join(project_root, "saved_agents", "ppo_lift", "lift_extended.zip"),
        ])
        
        for ckpt in search_paths:
            if os.path.exists(ckpt):
                try:
                    if ckpt.endswith(".zip"):
                        from stable_baselines3 import PPO
                        policy = PPO.load(ckpt)
                        self.logger.info(f"[{self.name}] Successfully loaded Stable-Baselines3 PPO policy from: {ckpt}")
                        return policy
                    else:
                        import torch
                        policy = torch.load(ckpt, map_location="cpu")
                        if hasattr(policy, "eval"):
                            policy.eval()
                        self.logger.info(f"[{self.name}] Successfully loaded PyTorch PPO policy from: {ckpt}")
                        return policy
                except Exception as exc:
                    self.logger.warning(f"[{self.name}] Error loading checkpoint {ckpt}: {exc}")
                    
        self.logger.info(f"[{self.name}] No active PPO checkpoint found yet in search paths. Using fallback controller until checkpoint is added.")
        return None

    def is_done(self, obs: dict) -> bool:
        if self.stage not in ["ppo_lift", "lift"]:
            return False
        return self.env.is_grasped(self.args.get("object", ""))

    def _get_action(self, obs: dict) -> np.ndarray:
        target_obj = self.args.get("object", "")
        body_id = self.env._resolve_body_id(target_obj)
        if body_id is None:
            return np.zeros(7)

        obj_pos = self.env._base_env.sim.data.body_xpos[body_id]
        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
        
        action = np.zeros(7)
        action[6] = -1.0 # open gripper by default during navigation
        
        # ── Stage 1: RRT Global Navigation to hover zone ───────────
        if self.stage == "follow_path":
            if not hasattr(self, "_wp_timer"):
                self._wp_timer = 0
            self._wp_timer += 1
            if self.current_wp_idx < len(self.path):
                target_pos = self.path[self.current_wp_idx]
                if np.linalg.norm(eef_pos - target_pos) < 0.04 or self._wp_timer > 30:
                    self.current_wp_idx += 1
                    self._wp_timer = 0
            else:
                self.stage = "descend"
                target_pos = eef_pos
                
        # ── Stage 2: Descend to Pre-Grasp Pose ─────────────────────
        if self.stage == "descend":
            target_pos = obj_pos.copy()
            target_pos[2] += self.descend_height
            if not hasattr(self, "_descend_timer"):
                self._descend_timer = 0
            self._descend_timer += 1
            
            xy_err = np.linalg.norm(eef_pos[:2] - target_pos[:2])
            z_err = abs(eef_pos[2] - target_pos[2])
            
            # Transition to PPO hand-off once aligned at pre-grasp height
            if (xy_err < 0.025 and z_err < 0.020) or self._descend_timer > 100:
                self.stage = "ppo_lift" if self.policy is not None else "grasp"
                self.grasp_timer = 0

        # ── Stage 3A: PPO Policy Execution for Grasp & Lift ───────
        if self.stage == "ppo_lift":
            if self.policy is not None:
                try:
                    rel_pos = eef_pos - obj_pos
                    gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
                    ppo_state = np.concatenate([rel_pos, gripper_qpos])
                    
                    if hasattr(self.policy, "predict"):
                        # Stable-Baselines3 interface
                        act, _ = self.policy.predict(ppo_state, deterministic=True)
                        action = np.array(act)
                    else:
                        # PyTorch model interface
                        import torch
                        with torch.no_grad():
                            t_obs = torch.as_tensor(ppo_state, dtype=torch.float32).unsqueeze(0)
                            act = self.policy(t_obs)
                            action = act.cpu().numpy().squeeze(0)
                    return action
                except Exception as exc:
                    self.logger.warning(f"[{self.name}] PPO prediction failed ({exc}), reverting to fallback grasp.")
                    self.stage = "grasp"

        # ── Stage 3B: Fallback Controller (if PPO model not loaded) ──
        if self.stage == "grasp":
            action[6] = 1.0 # close gripper
            self.grasp_timer += 1
            target_pos = obj_pos.copy()
            target_pos[2] += self.descend_height
            if self.grasp_timer > 40:
                self.stage = "lift"
                self._lift_timer = 0
                self._initial_obj_z = obj_pos[2]
                
        elif self.stage == "lift":
            target_pos = obj_pos.copy()
            target_pos[2] = self._initial_obj_z + self.hover_height
            action[6] = 1.0 # keep closed
            if not hasattr(self, "_lift_timer"):
                self._lift_timer = 0
            self._lift_timer += 1
            if self._lift_timer > 40 and (obj_pos[2] < self.env._table_height + 0.05):
                self.stage = "descend"
                self._descend_timer = 0
                action[6] = -1.0
        
        if self.stage == "follow_path":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -self.max_speed, self.max_speed)
        elif self.stage == "descend":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -0.40, 0.40)
        elif self.stage == "grasp":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -0.35, 0.35)
        elif self.stage == "lift":
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -0.35, 0.35)
            
        return action


@register_skill("MotionPlanningPlaceInBin")
class MotionPlanningPlaceInBinSkill(Skill):
    """
    Fallback motion planning controller to place an object in a bin.
    Uses TaskSpaceRRT to navigate to the bin, then opens the gripper.
    """

    def initialise(self):
        super().initialise()
        self.stage = "follow_path"
        self.hover_height = 0.18 # safely above container rim without glancing collision
        self.kp = 10.0
        self.max_speed = 1.0
        self.release_timer = 0
        
        self.path = []
        self.current_wp_idx = 0
        
        if self.env is None: return
        
        # 1. Get targets
        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
        
        target_container_pos = self.env._resolve_bin_pos(self.args.get("asset", "bin"))
        if target_container_pos is None:
            self.logger.warning(f"[{self.name}] Container position not found!")
            return
            
        # 2. Extract Obstacles (avoid other container during transit)
        target_asset = str(self.args.get("asset", "bin")).lower().strip()
        obstacles = []
        if target_asset in ["pot", "bowl"]:
            other_bin_pos = self.env._resolve_bin_pos("bin")
            if other_bin_pos is not None:
                obstacles.append({
                    'min': other_bin_pos - np.array([0.16, 0.16, 0.05]),
                    'max': other_bin_pos + np.array([0.16, 0.16, 0.20])
                })
        else:
            pot_pos = self.env._resolve_bin_pos("pot")
            if pot_pos is not None:
                obstacles.append({
                    'min': pot_pos - np.array([0.15, 0.18, 0.05]),
                    'max': pot_pos + np.array([0.15, 0.18, 0.20])
                })
            
        # 3. Plan Path
        planner = TaskSpaceRRT()
        goal_hover = target_container_pos.copy()
        goal_hover[2] += self.hover_height
        self.goal_hover = goal_hover
        self.path = planner.plan(eef_pos, goal_hover, obstacles)
        self.logger.info(f"[{self.name}] Generated path with {len(self.path)} waypoints")

    def load_policy(self):
        return None

    def is_done(self, obs: dict) -> bool:
        if self.stage != "release":
            return False
        if self.release_timer < 20:
            return False
        return self.env.is_in_bin(
            self.args.get("object", ""),
            self.args.get("asset", "bin"),
        )

    def _get_action(self, obs: dict) -> np.ndarray:
        if self.env is None: return np.zeros(7)
        
        eef_site_id = self.env._base_env.robots[0].eef_site_id["right"]
        eef_pos = self.env._base_env.sim.data.site_xpos[eef_site_id]
        
        action = np.zeros(7)
        action[6] = 1.0 # keep gripper closed by default
        target_pos = eef_pos
        
        if self.stage == "follow_path":
            if not hasattr(self, "_wp_timer"):
                self._wp_timer = 0
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
                
        elif self.stage == "release":
            action[6] = -1.0 # open gripper
            self.release_timer += 1
            if hasattr(self, "goal_hover"):
                target_pos = self.goal_hover
        
        if self.stage in ("follow_path", "release"):
            delta = target_pos - eef_pos
            action[:3] = np.clip(self.kp * delta, -self.max_speed, self.max_speed)
            
        return action