"""
JSON / PDDL plan -> py_trees tree builder.

Converts JSON tree specifications or PDDL plan strings into executable py_trees Behavior Trees.

Expected node shapes (JSON spec):
  {"type": "sequence" | "selector", "children": [ ... ], "memory": true}
  {"type": "action",    "name": "PickUp", "args": {"object": "can_1", ...}}
  {"type": "condition", "name": "VisualCheck", "args": {"true_situation": "..."}}
  {"type": "decorator", "decorator": "retry" | "inverter", "child": {...}, "num_attempts": 10}
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import py_trees

from .skills import SKILL_REGISTRY
from .conditions import CONDITION_REGISTRY

COMPOSITE_REGISTRY = {
    "sequence": py_trees.composites.Sequence,
    "selector": py_trees.composites.Selector,
}


def build_node(spec: Dict[str, Any], env=None, camera=None, vlm_query_fn=None) -> py_trees.behaviour.Behaviour:
    """Build a single py_trees node (and its children) from a dictionary specification."""
    node_type = spec["type"]

    if node_type in COMPOSITE_REGISTRY:
        composite_cls = COMPOSITE_REGISTRY[node_type]
        node = composite_cls(name=spec.get("name", node_type), memory=spec.get("memory", True))
        for child_spec in spec["children"]:
            node.add_child(build_node(child_spec, env=env, camera=camera, vlm_query_fn=vlm_query_fn))
        return node

    if node_type == "decorator":
        child = build_node(spec["child"], env=env, camera=camera, vlm_query_fn=vlm_query_fn)
        deco = spec["decorator"]
        if deco == "retry":
            return py_trees.decorators.Retry(
                name=spec.get("name", "retry"), child=child,
                num_failures=spec.get("num_attempts", 10),
            )
        if deco == "inverter":
            return py_trees.decorators.Inverter(name=spec.get("name", "inverter"), child=child)
        raise ValueError(f"Unknown decorator '{deco}'")

    if node_type == "action":
        skill_cls = SKILL_REGISTRY.get(spec["name"])
        if skill_cls is None:
            raise KeyError(f"Unregistered skill '{spec['name']}'. Known skills: {list(SKILL_REGISTRY)}")
        return skill_cls(name=spec["name"], args=spec.get("args", {}), env=env)

    if node_type == "condition":
        cond_cls = CONDITION_REGISTRY.get(spec["name"])
        if cond_cls is None:
            raise KeyError(f"Unregistered condition '{spec['name']}'. Known conditions: {list(CONDITION_REGISTRY)}")
        return cond_cls(name=spec["name"], args=spec.get("args", {}), camera=camera, vlm_query_fn=vlm_query_fn)

    raise ValueError(f"Unknown node type '{node_type}'")


def wrap_with_goal_check(
    main_sequence_spec: Dict[str, Any],
    goal_check_spec: Dict[str, Any],
    env=None,
    camera=None,
    vlm_query_fn=None,
    num_attempts: int = 10
) -> py_trees.behaviour.Behaviour:
    """
    Wrap main sequence with top-level goal check and retry loop:
        root = selector [
            GoalCheck,
            retry( sequence [ MAIN_SEQUENCE, GoalCheck ] )
        ]
    """
    goal_check_top = build_node(goal_check_spec, env=env, camera=camera, vlm_query_fn=vlm_query_fn)
    main_sequence = build_node(main_sequence_spec, env=env, camera=camera, vlm_query_fn=vlm_query_fn)
    goal_check_after = build_node(goal_check_spec, env=env, camera=camera, vlm_query_fn=vlm_query_fn)

    attempt = py_trees.composites.Sequence(name="attempt", memory=True)
    attempt.add_child(main_sequence)
    attempt.add_child(goal_check_after)

    retry = py_trees.decorators.Retry(name="retry_until_goal", child=attempt, num_failures=num_attempts)

    root = py_trees.composites.Selector(name="root", memory=False)
    root.add_child(goal_check_top)
    root.add_child(retry)
    return root


def pddl_plan_to_bt_spec(plan: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Convert a PDDL plan string (or list of action strings) into a Behavior Tree specification dictionary.

    Example input:
        "(pick red_can)\n(place red_can bin)"
    Example output spec:
        {
            "type": "sequence",
            "name": "pddl_plan_sequence",
            "memory": True,
            "children": [
                {"type": "action", "name": "PickUp", "args": {"object": "red_can"}},
                {"type": "action", "name": "PlaceInBin", "args": {"object": "red_can", "asset": "bin"}}
            ]
        }
    """
    if isinstance(plan, str):
        lines = plan.strip().split('\n')
    else:
        lines = plan

    children = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
            
        # Clean surrounding parentheses e.g. "(pick red_can)" -> "pick red_can"
        clean_line = line.lstrip("(").rstrip(")").strip()
        tokens = clean_line.split()
        if not tokens:
            continue
            
        action_name = tokens[0].lower()
        if action_name == "pick":
            obj_name = tokens[1] if len(tokens) > 1 else ""
            children.append({
                "type": "action",
                "name": "MotionPlanningPickUp",
                "args": {"object": obj_name}
            })
        elif action_name == "place":
            obj_name = tokens[1] if len(tokens) > 1 else ""
            asset_name = tokens[2] if len(tokens) > 2 else "bin"
            children.append({
                "type": "action",
                "name": "MotionPlanningPlaceInBin",
                "args": {"object": obj_name, "asset": asset_name}
            })
        else:
            # Generic action mapping fallback
            args = {f"arg{i}": tok for i, tok in enumerate(tokens[1:])}
            children.append({
                "type": "action",
                "name": tokens[0],
                "args": args
            })

    return {
        "type": "sequence",
        "name": "pddl_plan_sequence",
        "memory": True,
        "children": children
    }


def build_bt_from_pddl_plan(
    plan: Union[str, List[str]],
    env=None,
    camera=None,
    vlm_query_fn=None,
    goal_situation: Optional[str] = None,
    num_attempts: int = 10
) -> py_trees.behaviour.Behaviour:
    """
    Build an executable py_trees Behavior Tree directly from a PDDL plan.
    If goal_situation is provided, wraps the tree in a goal check retry loop.
    """
    main_spec = pddl_plan_to_bt_spec(plan)

    if goal_situation:
        goal_spec = {
            "type": "condition",
            "name": "GoalCheck",
            "args": {"true_situation": goal_situation}
        }
        return wrap_with_goal_check(
            main_sequence_spec=main_spec,
            goal_check_spec=goal_spec,
            env=env,
            camera=camera,
            vlm_query_fn=vlm_query_fn,
            num_attempts=num_attempts
        )

    return build_node(main_spec, env=env, camera=camera, vlm_query_fn=vlm_query_fn)