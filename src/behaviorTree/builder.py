"""
Behavior Tree Builder:
Compiles PDDL plans into executable py_trees Behavior Trees.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Union
import py_trees

from .skills import SKILL_REGISTRY
from .conditions import CONDITION_REGISTRY, PDDLGoalCheck


# ---------------------------------------------------------------------------
# PDDL Plan Parsing & Compilation
# ---------------------------------------------------------------------------
def parse_pddl_action(line: str) -> Optional[tuple[str, List[str]]]:
    """
    Parse a single PDDL plan action line.

    Examples:
        "(pick yellow_cube)"      -> ("pick", ["yellow_cube"])
        "0: (place cube pot)"     -> ("place", ["cube", "pot"])
        "; cost = 2 (unit cost)"  -> None
    """
    clean = line.strip()
    if not clean or clean.startswith(";"):  # Clean Fast Downward comments
        return None

    # Strip step index if present (e.g., "0: (pick obj)")
    if ":" in clean and clean.split(":", 1)[0].isdigit():
        clean = clean.split(":", 1)[1].strip()

    # Strip surrounding parentheses 
    clean = clean.lstrip("(").rstrip(")").strip()
    tokens = clean.split()
    if not tokens:
        return None

    # return the action name (tokens[0]) and the arguments (tokens[1:])
    return tokens[0].lower(), tokens[1:]


def pddl_plan_to_sequence(plan: Union[str, List[str]], env=None) -> py_trees.composites.Sequence:
    """
    Compile Fast Downward PDDL plan lines into an executable py_trees Sequence.

    Example:
        "(pick yellow_cube)
        (place yellow_cube pot)"
    Yields:
        Sequence [
            MotionPlanningPickUp(object='yellow_cube'),
            MotionPlanningPlaceInBin(object='yellow_cube', asset='pot')
        ]
    """
    lines = plan.strip().split("\n") if isinstance(plan, str) else plan
    sequence = py_trees.composites.Sequence(name="pddl_plan_sequence", memory=True)

    for line in lines:
        parsed = parse_pddl_action(line)
        if not parsed:
            continue

        action, args = parsed

        if action == "pick":
            obj = args[0] if args else ""
            skill_cls = SKILL_REGISTRY.get("pick", SKILL_REGISTRY.get("MotionPlanningPickUp"))
            if not skill_cls:
                raise KeyError("Action 'pick' not found in SKILL_REGISTRY")
            sequence.add_child(skill_cls(name=f"PickUp({obj})", args={"object": obj}, env=env))

        elif action == "place":
            obj = args[0] if args else ""
            target = args[1] if len(args) > 1 else "bin"
            skill_cls = SKILL_REGISTRY.get("place", SKILL_REGISTRY.get("MotionPlanningPlaceInBin"))
            if not skill_cls:
                raise KeyError("Action 'place' not found in SKILL_REGISTRY")
            sequence.add_child(
                skill_cls(name=f"PlaceInBin({obj}->{target})", args={"object": obj, "asset": target}, env=env)
            )

        elif action in SKILL_REGISTRY:
            skill_cls = SKILL_REGISTRY[action]
            kwargs = {f"arg{i}": a for i, a in enumerate(args)}
            sequence.add_child(skill_cls(name=action, args=kwargs, env=env))

        else:
            raise KeyError(f"Unknown PDDL action '{action}'. Registered skills: {list(SKILL_REGISTRY.keys())}")

    return sequence


# ---------------------------------------------------------------------------
# Goal-Guarded Wrapper
# ---------------------------------------------------------------------------
def wrap_with_goal_check(
    sequence: Optional[py_trees.behaviour.Behaviour] = None,
    env=None,
    num_attempts: int = 10,
    camera: Any = None,
    vlm_query_fn: Optional[Any] = None,
    goal_situation: Optional[str] = None,
    main_sequence: Optional[py_trees.behaviour.Behaviour] = None,
) -> py_trees.composites.Selector:
    """
    Wrap an action sequence in a reactive retry loop guarded by GoalCheck:

        root = Selector [
            GoalCheck_Pre  (check if goal is already met before acting),
            Retry( Sequence [ sequence, GoalCheck_Post ], num_failures=num_attempts )
        ]
    """
    seq = sequence or main_sequence
    if seq is None:
        raise ValueError("Must provide an action sequence to wrap_with_goal_check.")

    args = {"true_situation": goal_situation} if goal_situation else {}
    cond_cls = CONDITION_REGISTRY.get("GoalCheck", PDDLGoalCheck)

    goal_check_pre = cond_cls(name="GoalCheck_Pre", args=args, env=env, camera=camera, vlm_query_fn=vlm_query_fn)
    goal_check_post = cond_cls(name="GoalCheck_Post", args=args, env=env, camera=camera, vlm_query_fn=vlm_query_fn)

    attempt = py_trees.composites.Sequence(name="attempt", memory=True)
    attempt.add_child(seq)
    attempt.add_child(goal_check_post)

    retry = py_trees.decorators.Retry(name="retry_until_goal", child=attempt, num_failures=num_attempts)

    root = py_trees.composites.Selector(name="goal_guarded_root", memory=False)
    root.add_child(goal_check_pre)
    root.add_child(retry)
    return root


# ---------------------------------------------------------------------------
# Tree Visualization & Rendering
# ---------------------------------------------------------------------------
def render_bt(
    root: py_trees.behaviour.Behaviour,
    name: str = "behavior_tree",
    target_dir: str = "images",
) -> str:
    """
    Render a py_trees Behavior Tree as a Graphviz diagram (.dot, .png, .svg).

    Args:
        root: Root node of the Behavior Tree.
        name: Base filename without extension.
        target_dir: Directory where diagram files are saved.

    Returns:
        str: Absolute path to the generated PNG diagram.
    """
    os.makedirs(target_dir, exist_ok=True)
    py_trees.display.render_dot_tree(
        root=root,
        name=name,
        target_directory=target_dir,
    )
    png_path = os.path.abspath(os.path.join(target_dir, f"{name}.png"))
    print(f"[BehaviorTree] Saved graphical tree diagram to: {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
def build_bt_from_pddl_plan(
    plan: Union[str, List[str]],
    env=None,
    camera: Any = None,
    vlm_query_fn: Optional[Any] = None,
    goal_situation: Optional[str] = None,
    num_attempts: int = 10,
    wrap_goal_check: bool = False,
    render: bool = False,
    render_name: str = "behavior_tree",
    render_dir: str = "images",
    **kwargs,
) -> py_trees.behaviour.Behaviour:
    """
    Build an executable py_trees Behavior Tree directly from a PDDL plan.

    If `goal_situation` or `wrap_goal_check` is enabled, wraps the sequence
    in a reactive GoalCheck retry loop.
    If `render` is True, saves Graphviz .dot, .png, and .svg diagrams to `render_dir`.
    """
    sequence = pddl_plan_to_sequence(plan, env=env)

    if goal_situation or wrap_goal_check:
        root = wrap_with_goal_check(
            sequence=sequence,
            env=env,
            num_attempts=num_attempts,
            camera=camera,
            vlm_query_fn=vlm_query_fn,
            goal_situation=goal_situation,
        )
    else:
        root = sequence

    if render:
        render_bt(root=root, name=render_name, target_dir=render_dir)

    return root