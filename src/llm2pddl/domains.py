import os
import uuid
import subprocess

from pydantic import BaseModel, field_validator, ValidationInfo
from typing import List

class Type(BaseModel):
    """
    Type Validation class:
    (type name)
    """
    name: str
    parent: str = "object"

class Predicate(BaseModel):
    """
    Predicate Validation class:
    (:predicates 
        (predicate p1 p2 ...)
    )
    """
    name: str
    parameters: List[str]

class Action(BaseModel):
    """
    Action Validation class:
    (:action name
        :parameters (p1, p2, ...)
        :precondition (and ...)
        :effect (and ...)
    )
    """
    name: str
    parameters: List[str]
    preconditions: List[str]
    effects: List[str]

    @field_validator('parameters')
    @classmethod
    def validate_action_parameters(cls, params: List[str]) -> List[str]:
        for p in params:
            if not p.startswith('?'):
                raise ValueError(f"Parameter '{p}' is invalid. All parameters must start with '?'")
        return params

    @field_validator('preconditions', 'effects')
    @classmethod
    def validate_action_precondition_effects(cls, statements: List[str], info: ValidationInfo) -> List[str]:
        # 1. Grab the valid parameters already checked (e.g., ['?can', '?bin'])
        valid_params = info.data.get('parameters', [])
        raw_predicates = info.data.get('predicates', [])
        
        # Extract the base predicate names (e.g., "clear" from "(clear ?x)")
        valid_predicate_names = {p.name for p in raw_predicates}
        
        for statement in statements:
            clean_text = statement.replace('(', ' ').replace(')', ' ')
            words = clean_text.split()
            
            for word in words:
                if word.startswith('?'):
                    if word not in valid_params:
                        raise ValueError(f"Variable '{word}' in '{statement}' is not declared in parameters.")
                elif word not in ['and', 'not', 'or', 'imply', 'when', '=', '-', 'increase', 'decrease']:
                    # Check if the word matches a declared predicate name
                    if word not in valid_predicate_names:
                        raise ValueError(f"Term '{word}' in '{statement}' is not a declared predicate or operator.")

        return statements

class Domain(BaseModel):
    """
    Domain Validation class:
    (:domain name
        :requirements ()
        :types ()
        :predicates ()
        :action ()
    )
    """
    domain_name: str
    requirements: List[str]
    types: List[Type]
    predicates: List[Predicate]
    actions: List[Action]

    def to_pddl(self) -> str:
        pddl = f"(define (domain {self.domain_name})\n"
        pddl += f"(:requirements {' '.join([f':{r}' for r in self.requirements])})\n"
        pddl += f"(:types\n{' '.join([f'{t.name} - {t.parent}' for t in self.types])})\n"
        pddl += f"(:predicates\n"
        pddl += '\n'.join([f"  ({p.name} {' '.join(p.parameters)})" for p in self.predicates])
        pddl += "\n)\n"
        for a in self.actions:
            pddl += f"(:action {a.name}\n"
            pddl += f" :parameters ({' '.join(a.parameters)})\n"
            pddl += f" :precondition (and {' '.join(a.preconditions)})\n"
            pddl += f" :effect (and {' '.join(a.effects)})\n"
            pddl += ")\n"
        pddl += ")"
        return pddl

        
def postprocess(x):
    """Remove withe spaces"""
    return x.strip()

def get_random_temp_file_name(suffix='txt'):
    return os.path.join('/tmp', f"{str(uuid.uuid4())[:8]}.{suffix}")

def as_file(x, suffix='txt'):
    f_name = get_random_temp_file_name(suffix=suffix)
    with open(f_name, 'w') as f:
        f.write(x)
    return f_name

def read_and_remove_file(f_name):
    with open(f_name, 'r') as f:
        x = f.read()
    os.remove(f_name)
    return x


class PDDLenv:
    OPTIMAL_ALIAS = "seq-opt-fdss-1"
    SUB_OPTIMAL_ALIAS = "lama-first"

    def __init__(
            self, 
            fast_downward_path: str,
            time_limit: int,
            planning_algorithm: str = SUB_OPTIMAL_ALIAS
    ) -> None:
        self.fast_downward_path = fast_downward_path
        self.time_limit = time_limit
        self.planning_algorithm = planning_algorithm

    def search_plan(self, domain_pddl: str, problem_pddl: str):
        domain_pddl_path = as_file(domain_pddl)
        problem_pddl_path = as_file(problem_pddl)
        temp_plan_path = get_random_temp_file_name()
        temp_sas_path = get_random_temp_file_name()
        output = subprocess.run(
            [
                "python3",
                self.fast_downward_path,
                "--alias",
                self.planning_algorithm,
                "--search-time-limit",
                f"{self.time_limit}",
                "--plan-file",
                temp_plan_path,
                "--sas-file",
                temp_sas_path,
                domain_pddl_path,
                problem_pddl_path
            ],
            capture_output=True,
            text=True,
            universal_newlines=True,
        )

        search_output = output.stdout
        search_error = output.stderr
        read_and_remove_file(domain_pddl_path)
        read_and_remove_file(problem_pddl_path)
        if "Solution found." in search_output:
            plan = postprocess(read_and_remove_file(temp_plan_path))
            return plan, True, "Solution found."
        elif "Search stopped without finding a solution." in search_output:
            return None, True, "Generated PDDL domain is valid, but plan search stopped without finding a solution."
        elif "Time limit has been reached." in search_output:
            return None, True, "Generated PDDL domain is valid, but search Time limit has been reached."
        else:
            return None, False, search_error