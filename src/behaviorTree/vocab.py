"""
The "vocabulary" contract handed to llava: which leaf node names exist, what
arguments they take, and what they mean. This is the JSON analogue of the
MS repo's node_definitions_sample.json5, and it should be kept in lockstep
with skills.SKILL_REGISTRY / conditions.CONDITION_REGISTRY -- validate_plan()
below enforces that at runtime instead of just hoping the prompt is followed.
"""
from __future__ import annotations
from typing import Any, Dict

ACTIONS = {
    "PickUp": {
        "args": ["object", "asset"],
        "doc": "Grasp @object from @asset.",
    },
    "PlaceInBin": {
        "args": ["object", "asset"],
        "doc": "Place the held @object into @asset (a bin). "
               "Only valid after @object was picked up earlier in the plan.",
    },
    "ThrowAway": {
        "args": ["object", "asset"],
        "doc": "Discard the held @object into @asset. "
               "Only valid after @object was picked up earlier in the plan.",
    },
}

CONDITIONS = {
    "VisualCheck": {
        "args": ["true_situation"],
        "doc": "Ask the vision system whether @true_situation currently holds. "
               "Only use as the first child of a sequence.",
    },
    # GoalCheck is intentionally omitted -- the VLM should never author it,
    # it is injected by builder.wrap_with_goal_check().
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert interpreter of human instructions for a robot.
Given an instruction and a description of the scene, output ONLY a JSON object (no prose,
no markdown fences) with this exact shape:

{{
  "main_sequence": <tree node>,
  "ultimate_goal": "<a visually checkable statement of the completed goal>"
}}

A <tree node> is one of:
  {{"type": "sequence", "children": [<tree node>, ...]}}
  {{"type": "selector",  "children": [<tree node>, ...]}}
  {{"type": "action",    "name": "<one of: {action_names}>", "args": {{...}}}}
  {{"type": "condition", "name": "VisualCheck", "args": {{"true_situation": "..."}}}}

Rules:
1. Use ONLY the action names listed above, with EXACTLY the argument keys they require.
2. Every argument value must refer to an object/asset/location present in the scene description.
3. For an if/else instruction, put the condition as the FIRST child of a sequence nested
   inside a selector, with the "else" branch as the selector's second child.
4. Output must be valid JSON. No comments, no trailing commas.

Actions:
{action_docs}

Conditions:
{condition_docs}
"""


def _fmt_docs(spec: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for name, info in spec.items():
        lines.append(f"- {name}({', '.join(info['args'])}): {info['doc']}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        action_names=", ".join(ACTIONS.keys()),
        action_docs=_fmt_docs(ACTIONS),
        condition_docs=_fmt_docs(CONDITIONS),
    )


def build_user_prompt(scene_description: Dict[str, Any], instruction: str) -> str:
    import json
    return (
        f'Scene:\n"""\n{json.dumps(scene_description, indent=2)}\n"""\n\n'
        f'Instruction: "{instruction}"\n\n'
        "Respond with the JSON object described above and nothing else."
    )


def validate_plan(node: Dict[str, Any]) -> None:
    """Walk a parsed plan and raise if it references an unknown skill/condition
    or is missing required args -- catches VLM hallucinations before build_node()."""
    t = node.get("type")
    if t in ("sequence", "selector"):
        for child in node.get("children", []):
            validate_plan(child)
    elif t == "action":
        spec = ACTIONS.get(node.get("name"))
        if spec is None:
            raise ValueError(f"Unknown action '{node.get('name')}'")
        missing = set(spec["args"]) - set(node.get("args", {}).keys())
        if missing:
            raise ValueError(f"Action '{node['name']}' missing args: {missing}")
    elif t == "condition":
        spec = CONDITIONS.get(node.get("name"))
        if spec is None:
            raise ValueError(f"Unknown condition '{node.get('name')}'")
    elif t == "decorator":
        validate_plan(node["child"])
    else:
        raise ValueError(f"Unknown node type '{t}'")