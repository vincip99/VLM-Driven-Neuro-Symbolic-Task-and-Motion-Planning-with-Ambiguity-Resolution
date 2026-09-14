"""
Leaf "Condition" nodes:
  - VisualCheck: Query the VLM to evaluate if a visual situation is true.
  - GoalCheck: Same mechanism, reserved for top-level tree goal checks.

Both ask the VLM a yes/no question grounded in a camera frame.
"""
from __future__ import annotations

from typing import Any, Callable, Optional
import py_trees

CONDITION_REGISTRY: dict[str, type[VisualCheck]] = {}


def register_condition(name: str):
    """Decorator to register condition classes by name in CONDITION_REGISTRY."""
    def decorator(cls: type[VisualCheck]):
        CONDITION_REGISTRY[name] = cls
        return cls
    return decorator


@register_condition("VisualCheck")
class VisualCheck(py_trees.behaviour.Behaviour):
    """Behavior Tree node that queries the VLM for a visual condition check."""

    def __init__(
        self,
        name: str,
        args: dict[str, Any],
        camera: Any = None,
        vlm_query_fn: Optional[Callable[[Any, str], str]] = None,
    ):
        super().__init__(name=name)
        self.situation = args.get("true_situation", args.get("what_to_check", ""))
        self.camera = camera
        self.vlm_query_fn = vlm_query_fn

    def update(self) -> py_trees.common.Status:
        if self.vlm_query_fn is None:
            self.logger.error(f"[{self.name}] Missing vlm_query_fn")
            return py_trees.common.Status.FAILURE

        frame = self.camera.get_frame() if self.camera is not None else None
        prompt = f"Answer only 'yes' or 'no'. Is the following true: {self.situation}"
        answer = self.vlm_query_fn(frame, prompt).strip().lower()

        is_success = answer.startswith("y")
        self.logger.debug(f"[{self.name}] '{self.situation}' -> {answer} ({is_success})")

        return (
            py_trees.common.Status.SUCCESS
            if is_success
            else py_trees.common.Status.FAILURE
        )


@register_condition("GoalCheck")
class GoalCheck(VisualCheck):
    """
    Reserved node type for top-level goal checks (builder.wrap_with_goal_check).
    Inherits all execution logic directly from VisualCheck.
    """
    pass