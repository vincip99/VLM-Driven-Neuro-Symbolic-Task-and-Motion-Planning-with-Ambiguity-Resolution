"""
Action extractor for Behavior Trees.
Extracts actions and parameters directly from a PDDL domain file.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional


def extract_actions_from_domain(domain_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Parse a PDDL domain file and extract action names and their parameters.

    Returns:
        Dict[str, List[str]]: e.g. {'pick': ['ob'], 'place': ['ob', 'pos']}
    """
    if domain_path is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(curr_dir, "..", ".."))
        candidates = [
            os.path.join(repo_root, "domain.pddl"),
            os.path.join(repo_root, "domains", "manipulation", "domain.pddl"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                domain_path = candidate
                break

    if not domain_path or not os.path.isfile(domain_path):
        return {}

    with open(domain_path, "r") as f:
        text = f.read()

    # Remove comments
    text = re.sub(r";.*", "", text)

    actions: Dict[str, List[str]] = {}
    for block in text.split("(:action")[1:]:
        action_name = block.strip().split()[0]
        if ":parameters" in block:
            params_part = block.split(":parameters")[1].split(")")[0].replace("(", "")
            params = [p.replace("?", "") for p in params_part.split() if p.startswith("?")]
        else:
            params = []
        actions[action_name] = params

    return actions


def validate_plan(node: dict, actions: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Validate that action nodes in a Behavior Tree plan match extracted domain actions.
    """
    if actions is None:
        actions = ACTIONS

    node_type = node.get("type")
    if node_type in ("sequence", "selector"):
        for child in node.get("children", []):
            validate_plan(child, actions)
    elif node_type == "action":
        name = node.get("name")
        valid_names = set(actions.keys()) | {k.lower() for k in actions.keys()}
        if name and name.lower() not in valid_names and name not in actions:
            raise ValueError(f"Unknown action '{name}'. Known actions from domain: {list(actions.keys())}")


# Precompile actions on import
ACTIONS: Dict[str, List[str]] = extract_actions_from_domain()