import json
import warnings
from typing import List, Tuple, Optional
from PIL import Image
from pydantic import BaseModel
from .vlm_chat import VlmChat
from . import ASSETS_DIR
from src.llm2pddl import problem_translation_prompt, domain_translation_prompt
from src.llm2pddl.prompt import ONE_SHOT_INIT_PROMPT_TEMPLATE, BLOCKS_WORLD_EXAMPLE
from src.llm2pddl.domains import Domain
from src.llm2pddl.problem import Problem


class ObjGrounding(BaseModel):
    """
    Define the schema of grounded objects
    """
    task_objects: List[str]

class RobotSkill(BaseModel):
    """
    This class defines the schema of a robot skill.
    """
    skill: str
    target_object: ObjGrounding
    goal: str

class RobotTask(BaseModel):
    """
    This class defines the schema of a robot task.
    """
    reasoning: str
    skills: List[RobotSkill]

class AmbiguityReasoning(BaseModel):
    task_ambiguous: bool
    explanation: str
    clarifying_question: str


class AmbresStructured(VlmChat):
    def __init__(self, vlm_name="qwen2.5vl:3b"):
        super().__init__(vlm_name=vlm_name)

    def add_message(self, role: str, content:str):
        super().add_message(role, content)

    def handle_query(self, task_description: str, image) -> dict:
        """
        Handle the query from the user. Accepts a single PIL image or a list of PIL images.
        """
        # set the image(s) and construct the message for the vlm query
        if isinstance(image, list):
            self.set_image(image)
            img_desc = f"Please analyze the attached {len(image)} images and extract the objects. "
        else:
            self.set_image([image])
            img_desc = "Please analyze the attached image and extract the objects. "

        user_text = (
            "Task Description: " + task_description + "\n" +
            img_desc +
            "Remember to keep all adjectives and colors exactly as I said them!"
        )
        self.add_message("user", user_text)
        text_out = self.inference(
            self.messages
        )
        # get ai reply and save to self.messages
        self.add_message("assistant", text_out)
        task_objects = json.loads(self.clean_json(text_out))["task_objects"]

        # ambiguity resolution
        self.add_message("user", "Is the task ambiguous?")
        text_out = self.inference(
            self.messages,
            do_sample=False
        )
        # get ai reply (claryfing question) and save to self.messages
        self.add_message("assistant", text_out)

        try:
            data_out = json.loads(self.clean_json(text_out))
        except json.decoder.JSONDecodeError:
            warnings.warn("LLM Output broken, will output default response.")
            data_out = {"task_ambiguous": False, "clarifying_question": ""}
        task_ambiguous = data_out["task_ambiguous"]

        clarifying_question = data_out["clarifying_question"] if task_ambiguous else ""
        output = {
            "task_objects": task_objects,
            "task_ambiguous": task_ambiguous,
            "clarifying_question": clarifying_question,
        }
        return output

    def handle_query_dict(self, input_data: dict) -> dict:
        task_description: str = input_data["task_description"]
        image = Image.open(input_data["image_path"])
        image = image.reduce(4)  # Downsample
        output = self.handle_query(task_description, image)
        return output

    def handle_response(self, response: str) -> dict:
        """
        Handle the response from the VLM.
        """
        self.add_message("user", response)
        text_out = self.inference(
            self.messages
        )
        self.add_message("assistant", text_out)
        task_objects = json.loads(self.clean_json(text_out))["task_objects"]
        output = {
            "task_objects": task_objects
        }
        return output

    def generate_problem_pddl(self, target_domain_nl: str, target_problem_nl: str, target_problem_template_pddl: str, target_domain_pddl: str = None) -> str:
        """
        Generate PDDL problem given the domain natural language, the finalized task description, 
        and the PDDL problem template.
        """
        system_message, few_shot_messages, user_input = problem_translation_prompt(
            target_domain_nl=target_domain_nl,
            target_problem_nl=target_problem_nl,
            target_problem_template_pddl=target_problem_template_pddl,
            target_domain_pddl=target_domain_pddl
        )
        
        pddl_messages = [
            {"role": "system", "content": system_message},
            *few_shot_messages,
            {"role": "user", "content": user_input}
        ]

        # Use inference with no sampling to ensure deterministic PDDL generation
        text_out = self.inference(pddl_messages, do_sample=False)
        return self.clean_vlm_code(text_out, "pddl")

    def generate_domain_pddl(
        self,
        context_domain_nl: str,
        context_problem_nl: str,
        context_domain_template_pddl: str,
        context_domain_pddl: str,
        target_domain_nl: str,
        target_problem_nl: str,
        target_domain_template_pddl: str
    ) -> str:
        """
        Generate PDDL domain file given natural language domain/problem descriptions and templates.
        """
        system_message, few_shot_messages, user_input = domain_translation_prompt(
            context_domain_nl=context_domain_nl,
            context_problem_nl=context_problem_nl,
            context_domain_template_pddl=context_domain_template_pddl,
            context_domain_pddl=context_domain_pddl,
            target_domain_nl=target_domain_nl,
            target_problem_nl=target_problem_nl,
            target_domain_template_pddl=target_domain_template_pddl
        )
        
        domain_messages = [
            {"role": "system", "content": system_message},
            *few_shot_messages,
            {"role": "user", "content": user_input}
        ]

        text_out = self.inference(domain_messages, do_sample=False)
        return self.clean_vlm_code(text_out, "pddl")

    def generate_pddl_json(self, task_description: str, task_objects: List[str]) -> Tuple[Domain, Problem]:
        """
        Generates Domain and Problem JSON objects using a few-shot example to enforce strict schema adherence.
        """
        # 1. The System Prompt (with explicit warnings about previous mistakes)
        json_system_prompt = (
            "You are an expert PDDL generator. Output ONLY a valid JSON object. "
            "Do not include any conversational text or explanations outside the JSON.\n\n"
            "The JSON must follow this exact structure:\n"
            "{\n"
            '  "domain": {\n'
            '    "domain_name": "...",\n'
            '    "requirements": [":strips"],\n'
            '    "types": [],\n'
            '    "predicates": [{"name": "predicate_name", "parameters": ["?x"]}],\n'
            '    "actions": [{"name": "action_name", "parameters": ["?ob"], "preconditions": ["(predicate_name ?ob)"], "effects": ["(...)"]}]\n'
            '  },\n'
            '  "problem": { ... }\n'
            "}"
        )

        # 2. The Few-Shot Example (User asks for Blocks World)
        few_shot_user = (
            "Task: Stack block A on block B.\n"
            "Grounded Objects: [\"block A\", \"block B\"]\n"
            "Generate the PDDL Domain and Problem JSON objects."
        )

        # 3. The Perfect Output (Assistant provides the perfectly formatted JSON)
        few_shot_assistant = """{
          "domain": {
            "domain_name": "blocksworld-4ops",
            "requirements": [":strips"],
            "types": [],
            "predicates": [
              {"name": "clear", "parameters": ["?x"]},
              {"name": "on-table", "parameters": ["?x"]},
              {"name": "arm-empty", "parameters": []},
              {"name": "holding", "parameters": ["?x"]},
              {"name": "on", "parameters": ["?x", "?y"]}
            ],
            "actions": [
              {
                "name": "pickup",
                "parameters": ["?ob"],
                "preconditions": ["(clear ?ob)", "(on-table ?ob)", "(arm-empty)"],
                "effects": ["(holding ?ob)", "(not (clear ?ob))", "(not (on-table ?ob))", "(not (arm-empty))"]
              },
              {
                "name": "putdown",
                "parameters": ["?ob"],
                "preconditions": ["(holding ?ob)"],
                "effects": ["(clear ?ob)", "(arm-empty)", "(on-table ?ob)", "(not (holding ?ob))"]
              },
              {
                "name": "stack",
                "parameters": ["?ob", "?underob"],
                "preconditions": ["(clear ?underob)", "(holding ?ob)"],
                "effects": ["(arm-empty)", "(clear ?ob)", "(on ?ob ?underob)", "(not (clear ?underob))", "(not (holding ?ob))"]
              },
              {
                "name": "unstack",
                "parameters": ["?ob", "?underob"],
                "preconditions": ["(on ?ob ?underob)", "(clear ?ob)", "(arm-empty)"],
                "effects": ["(holding ?ob)", "(clear ?underob)", "(not (on ?ob ?underob))", "(not (clear ?ob))", "(not (arm-empty))"]
              }
            ]
          },
          "problem": {
            "problem_name": "stack-a-on-b",
            "domain_name": "blocksworld-4ops",
            "objects": [
              {"name": "block_A", "type": "object"},
              {"name": "block_B", "type": "object"}
            ],
            "init": [
              {"name": "clear", "parameters": ["block_A"]},
              {"name": "on-table", "parameters": ["block_A"]},
              {"name": "clear", "parameters": ["block_B"]},
              {"name": "on-table", "parameters": ["block_B"]},
              {"name": "arm-empty", "parameters": []}
            ],
            "goal": [
              {"name": "on", "parameters": ["block_A", "block_B"]}
            ]
          }
        }"""

        # 4. The Actual Request
        user_prompt = (
            f"Task: {task_description}\n"
            f"Grounded Objects: {json.dumps(task_objects)}\n"
            "You are an expert PDDL generator. You must return a single valid JSON object with exactly two top-level keys: 'domain' and 'problem'.\n\n"
            "Output strictly valid JSON. Always use double quotes for all dictionary keys and string values. Never use single quotes (')"
            "CRITICAL RULES:\n"
            "1. Use EXACTLY the keys defined in the schema. Do not shorten 'domain_name' to 'name'.\n"
            "**Prerequisite Step**: Before writing your actions, you MUST declare every predicate used in those actions within the domain's 'predicates' array.\n"
            "2. Preconditions and effects MUST be a list of raw PDDL strings (e.g. [\"(on ?x ?y)\", \"(not (clear ?y))\"]). DO NOT output nested dictionaries for effects.\n"
            "3. Predicate Constraint: Actions can ONLY use predicates that are explicitly defined in your 'predicates' list. Do not invent new physical states or predicates.\n"
            "4. Variable Constraint: Every single variable (any word starting with ?) used in an action's 'preconditions' or 'effects' MUST be declared in that action's 'parameters' list.\n\n"
            f"Schema for Domain: {Domain.model_json_schema()}\n\n"
            f"Schema for Problem: {Problem.model_json_schema()}"
        )

        # 5. Build the message history
        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": few_shot_user},
            {"role": "assistant", "content": few_shot_assistant},
            {"role": "user", "content": user_prompt}
        ]

        # Call VLM with retry logic
        domain_obj, problem_obj = generate_with_retry(self, messages)

        # Run your cross-validation (make sure validate_problem_against_domain is imported/defined)
        validation_errors = validate_problem_against_domain(domain_obj, problem_obj)
        return domain_obj, problem_obj

    def generate_problem_json(
        self, 
        initial_state_desc: str, 
        task_description: str, 
        task_objects: List[str], 
        domain_pddl: str,
        target_locations: Optional[List[str]] = None
    ) -> Problem:
        """
        Generates Problem JSON object given a predefined PDDL domain definition.
        """
        json_system_prompt = (
            "You are an expert PDDL generator. Output ONLY a single valid JSON object. "
            "Do not include any conversational text, markdown notes, or explanations outside the JSON.\n\n"
            "The JSON must follow this exact structure:\n"
            "{\n"
            '  "problem": {\n'
            '    "problem_name": "...",\n'
            '    "domain_name": "...",\n'
            '    "objects": [{"name": "...", "type": "..."}],\n'
            '    "init": [{"name": "...", "parameters": ["..."]}],\n'
            '    "goal": [{"name": "...", "parameters": ["..."]}]\n'
            '  }\n'
            "}"
        )

        few_shot_user = (
            "Domain PDDL:\n"
            "(define (domain manipulation)\n"
            "  (:requirements :strips :typing)\n"
            "  (:types robot obj location)\n"
            "  (:predicates (holding ?ob) (on-table ?ob) (on ?ob ?pos))\n"
            "  (:action pick :parameters (?ob - obj) :precondition (and (not (holding ?ob)) (on-table ?ob)) :effect (and (holding ?ob) (not (on-table ?ob))))\n"
            "  (:action place :parameters (?ob - obj ?pos - location) :precondition (and (holding ?ob)) :effect (and (not (holding ?ob)) (on ?ob ?pos)))\n"
            ")\n\n"
            "Initial State Description: Both can_A and cube_B are resting on the table. The robot's arm is empty.\n"
            "Goal Task: Place can_A and cube_B into the bin.\n"
            "Grounded Objects: [\"can_A\", \"cube_B\"]\n"
            "Target Locations: [\"bin\"]\n"
            "Generate the PDDL Problem JSON object."
        )

        few_shot_assistant = """{
  "problem": {
    "problem_name": "manipulation-task",
    "domain_name": "manipulation",
    "objects": [
      {"name": "can_A", "type": "obj"},
      {"name": "cube_B", "type": "obj"},
      {"name": "bin", "type": "location"}
    ],
    "init": [
      {"name": "on-table", "parameters": ["can_A"]},
      {"name": "on-table", "parameters": ["cube_B"]}
    ],
    "goal": [
      {"name": "on", "parameters": ["can_A", "bin"]},
      {"name": "on", "parameters": ["cube_B", "bin"]}
    ]
  }
}"""

        # Convert spaces to underscores in task objects to avoid PDDL syntax errors
        safe_task_objects = [obj.strip().replace(" ", "_") for obj in task_objects]
        
        # Inferred target locations if not explicitly supplied
        if target_locations is None:
            target_locations = []
            for candidate in ["sorting_bin", "pot", "bowl", "bin"]:
                if candidate in task_description.lower().replace(" ", "_"):
                    target_locations.append(candidate)
            if not target_locations:
                target_locations = ["sorting_bin"]
        safe_locations = [loc.strip().replace(" ", "_") for loc in target_locations]

        user_prompt = (
            f"Domain PDDL:\n{domain_pddl}\n\n"
            f"Initial State Description: {initial_state_desc}\n"
            f"Goal Task: {task_description}\n"
            f"Grounded Manipulable Objects (type 'obj'): {json.dumps(safe_task_objects)}\n"
            f"Target Locations / Receptacles (type 'location'): {json.dumps(safe_locations)}\n"
            "You are an expert PDDL generator. You must return a single valid JSON object with a single top-level key: 'problem'.\n\n"
            "Output strictly valid JSON. Always use double quotes for all dictionary keys and string values. Never use single quotes (').\n"
            "CRITICAL RULES:\n"
            "1. Every single entity appearing in ':init' or ':goal' MUST be declared in the 'objects' list.\n"
            "2. Manipulable items (e.g. red_can, blue_can, yellow_cube) MUST be declared in 'objects' with type 'obj'.\n"
            "3. Target receptacles and containers (e.g. sorting_bin, pot, bin, bowl) MUST be declared in 'objects' with type 'location'. Do NOT use the type word 'location' as an object name.\n"
            "4. ONLY use predicates explicitly defined in the provided Domain PDDL. If a relationship like 'in' is requested, map it to 'on' with arguments: (on <obj> <location>).\n"
            "5. Ensure all object and location names use underscores instead of spaces (e.g., 'red_can', 'sorting_bin').\n"
            "6. Make sure the 'domain_name' matches the name in the Domain PDDL exactly.\n"
            "7. The ':goal' block must ONLY describe the FINAL physical state after all tasks are completed. Do NOT include intermediate action steps like 'holding' in the goal.\n\n"
            f"Schema for Problem: {Problem.model_json_schema()}"
        )

        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": few_shot_user},
            {"role": "assistant", "content": few_shot_assistant},
            {"role": "user", "content": user_prompt}
        ]

        problem_obj = generate_problem_with_retry(self, messages, domain_pddl)
        
        return problem_obj




class ambresFewShotPrompt(AmbresStructured):
    def __init__(self, vlm_name="qwen2.5vl:3b"):
        super().__init__(vlm_name=vlm_name)

    def reset_chat(self):
        """Use a System Prompt to teach the model, rather than fake history."""
        self.messages = []
        
        system_instruction = """You are the vision-language brain for a robot. 
Your job is to look at the user's task, extract ONLY the specific objects the robot needs to interact with, and check if the environment makes the task ambiguous.

CRITICAL RULES:
1. ONLY list objects that are explicitly part of the user's command. DO NOT list background items.
2. PRESERVE ADJECTIVES: You MUST keep all colors, sizes, and descriptive words exactly as the user wrote them. (e.g., If the user says "red cup", output "red cup", NOT just "cup").
3. If the task is ambiguous, your clarifying question MUST specifically mention the objects causing the confusion (e.g. "There are two cups, which one?").

EXAMPLES OF HOW YOU MUST THINK:

[Example 1 - Ambiguous]
User: NEW TASK DESCRIPTION: Put the banana in the drawer.
Please analyze the attached image and extract the objects. Keep adjectives.
Assistant: {"task_objects": ["banana", "drawer"]}
User: Is the task ambiguous?
Assistant: {"task_ambiguous": true, "explanation": "There are multiple drawers visible.", "clarifying_question": "There are three drawers. Which one do you mean?"}
User: The middle drawer.
Assistant: {"task_objects": ["banana", "middle drawer"]}

[Example 2 - Clear]
User: NEW TASK DESCRIPTION: Put the green mug in the microwave.
Please analyze the attached image and extract the objects. Keep adjectives.
Assistant: {"task_objects": ["green mug", "microwave"]}
User: Is the task ambiguous?
Assistant: {"task_ambiguous": false, "explanation": "There is only one microwave visible.", "clarifying_question": ""}

INSTRUCTIONS: 
Look at the NEW image and answer based ONLY on the NEW task description provided by the user below."""

        # Set this as the system prompt to anchor the model's behavior
        self.messages.append({"role": "system", "content": system_instruction})
        return


def validate_problem_against_domain(domain: Domain, problem: Problem) -> List[str]:
    """
    Checks if all predicates used in the problem exist in the domain definition.
    Returns a list of error messages (empty if valid).
    """
    # 1. Gather all allowed predicate names from the Domain
    valid_predicates = {pred.name for pred in domain.predicates}
    
    # Standard PDDL built-ins (like equality) are always allowed
    valid_predicates.add("=")

    errors = []

    # 2. Check all initial state predicates
    for pred in problem.init:
        if pred.name not in valid_predicates:
            errors.append(f"Hallucinated predicate in :init -> '{pred.name}' is not in Domain predicates!")

    # 3. Check all goal state predicates
    for pred in problem.goal:
        if pred.name not in valid_predicates:
            errors.append(f"Hallucinated predicate in :goal -> '{pred.name}' is not in Domain predicates!")

    return errors

def generate_problem_with_retry(self, messages: list, domain_pddl: str, max_retries: int = 5):
    import re
    # Extract valid predicates from domain_pddl string
    valid_predicates = {"="}
    pred_match = re.search(r'\(:predicates(.*?)(?:\(:|\)$)', domain_pddl, re.DOTALL | re.IGNORECASE)
    if pred_match:
        preds = re.findall(r'\(\s*([a-zA-Z0-9_\-]+)', pred_match.group(1))
        valid_predicates.update(preds)

    # Extract valid types from domain_pddl string
    valid_types = set() 
    type_match = re.search(r'\(:types(.*?)(?:\(:|\)$)', domain_pddl, re.DOTALL | re.IGNORECASE)
    if type_match:
        types_found = re.findall(r'([a-zA-Z0-9_\-]+)', type_match.group(1))
        valid_types.update(types_found)
    
    # If no types were explicitly declared, allow "object" as the fallback
    if not valid_types:
        valid_types.add("object")

    for attempt in range(max_retries):
        # 1. Generate text from LLM
        text_out = self.inference(messages)
        
        try:
            # 2. Parse and validate
            cleaned_json = json.loads(self.clean_json(text_out))
            problem_obj = Problem.model_validate(cleaned_json["problem"])
            
            # Auto-heal: If an undeclared entity is used in :init or :goal, auto-declare it with inferred type
            declared_objects = {obj.name for obj in problem_obj.objects}
            known_locations = {"sorting_bin", "bin", "pot", "bowl", "box", "tray", "table"}
            from src.llm2pddl.problem import PDDLObject
            for pred in list(problem_obj.init) + list(problem_obj.goal):
                for idx, param in enumerate(pred.parameters):
                    clean_p = param.strip().replace(" ", "_")
                    if clean_p and clean_p not in declared_objects and clean_p not in valid_types:
                        if clean_p in known_locations or (pred.name == "on" and idx == 1):
                            inferred_type = "location" if "location" in valid_types else "object"
                        else:
                            inferred_type = "obj" if "obj" in valid_types else "object"
                        problem_obj.objects.append(PDDLObject(name=clean_p, type=inferred_type))
                        declared_objects.add(clean_p)

            # Validate predicates, types, and undeclared objects
            errors = []
            for obj in problem_obj.objects:
                if obj.type not in valid_types:
                    errors.append(f"Hallucinated type for object '{obj.name}' -> '{obj.type}' is not in Domain types! Valid types are: {list(valid_types)}")
            for pred in problem_obj.init:
                if pred.name not in valid_predicates:
                    errors.append(f"Hallucinated predicate in :init -> '{pred.name}' is not in Domain predicates! Valid predicates are: {list(valid_predicates)}")
                for param in pred.parameters:
                    if param not in declared_objects:
                        errors.append(f"Undeclared object in :init -> '{param}' used in predicate '{pred.name}' but missing from :objects list!")
            for pred in problem_obj.goal:
                if pred.name not in valid_predicates:
                    errors.append(f"Hallucinated predicate in :goal -> '{pred.name}' is not in Domain predicates! Valid predicates are: {list(valid_predicates)}")
                for param in pred.parameters:
                    if param not in declared_objects:
                        errors.append(f"Undeclared object in :goal -> '{param}' used in predicate '{pred.name}' but missing from :objects list!")
            
            if errors:
                raise ValueError("PDDL Validation Failed:\n" + "\n".join(errors))
            
            # If everything passes, return the valid object
            return problem_obj
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                raise e # Stop if we run out of retries
                
            # 3. Feed the exact error back into the conversation history with prescriptive guidance
            messages.append({"role": "assistant", "content": text_out})
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous output caused this validation error:\n{e}\n\n"
                    "Please fix the PDDL problem JSON. Remember:\n"
                    "1. Every single object or receptacle used in :init or :goal MUST be declared in 'objects'.\n"
                    "2. Use type 'obj' for manipulable items and 'location' for target receptacles (e.g. sorting_bin, pot).\n"
                    "3. Do NOT use the type word 'location' as an object name.\n"
                    "4. Keep predicate names exact (e.g. 'on-table', 'on')."
                )
            })

def generate_with_retry(self, messages: list, max_retries: int = 5):
    for attempt in range(max_retries):
        # 1. Generate text from LLM
        text_out = self.inference(messages)
        
        try:
            # 2. Parse and validate
            cleaned_json = json.loads(self.clean_json(text_out))
            domain_obj = Domain.model_validate(cleaned_json["domain"])
            problem_obj = Problem.model_validate(cleaned_json["problem"])
            
            # If everything passes, return the valid objects
            return domain_obj, problem_obj
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                raise e # Stop if we run out of retries
                
            # 3. Feed the exact Pydantic error back into the conversation history
            messages.append({"role": "assistant", "content": text_out})
            messages.append({
                "role": "user",
                "content": f"Your previous output caused this validation error:\n{e}\n\nPlease fix the PDDL domain to use generic variables (starting with ?) in actions and keep grounded objects out of the domain blueprint."
            })