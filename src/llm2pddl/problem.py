from pydantic import BaseModel, field_validator
from typing import List

from src.llm2pddl.domains import Predicate

class PDDLObject(BaseModel):
    name: str
    type: str
    
    @field_validator("name", mode="before")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        return v.strip().replace(" ", "_")

class Problem(BaseModel):
    problem_name: str
    domain_name: str
    objects: List[PDDLObject]
    init: List[Predicate]
    goal: List[Predicate]

    def _format_predicate(self, pred: Predicate) -> str:
        if pred.parameters:
            params = " " + " ".join(p.replace(" ", "_") for p in pred.parameters)
        else:
            params = ""
        return f"({pred.name}{params})"
        
    def to_pddl(self) -> str:
        pddl = f"(define (problem {self.problem_name})\n"
        pddl += f"  (:domain {self.domain_name})\n"
        pddl += "  (:objects\n"
        for obj in self.objects:
            pddl += f"    {obj.name} - {obj.type}\n"
        pddl += "  )\n"
        pddl += "  (:init\n"
        for pred in self.init:
            pddl += f"    {self._format_predicate(pred)}\n"
        pddl += "  )\n"
        pddl += "  (:goal (and\n"
        for pred in self.goal:
            pddl += f"    {self._format_predicate(pred)}\n"
        pddl += "  ))\n"
        pddl += ")\n"
        return pddl
