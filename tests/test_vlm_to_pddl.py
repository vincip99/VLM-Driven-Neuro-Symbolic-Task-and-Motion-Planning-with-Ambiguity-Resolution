import os
from datetime import datetime
import cv2
import numpy as np
import robosuite as suite

# Adjust this import path based on where your classes are saved
from src.ambiguityres.vlm_model import ambresFewShotPrompt, AmbresStructured
from src.llm2pddl.domains import PDDLenv

def main():
    # 1. Setup
    print("Initializing vision pipeline (Qwen)...")
    vision_pipeline = ambresFewShotPrompt(vlm_name="qwen2.5vl:3b")
    vision_pipeline.reset_chat() 

    print("Initializing reasoning pipeline (Llama 3)...")
    reasoning_pipeline = AmbresStructured(vlm_name="llama3:latest")

    # Updated to match the objects in TaskSorting (red_can, sorting_bin)
    initial_task = "Put the red can in the sorting bin, then put the yellow cube in the pot. Then take the blue can and put it in the sorting bin."
    
    print("Initializing Robosuite Environment (TaskSorting)...")
    env = suite.make(
        env_name="TaskSorting",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["top_down_vlm", "frontview"],
        camera_heights=512,
        camera_widths=512,
    )
    obs = env.reset()
    
    rgb_top_down = np.flipud(obs["top_down_vlm_image"])
    image_path = os.path.abspath("test_vlm2pddl_camera_top_down.jpg")
    cv2.imwrite(image_path, cv2.cvtColor(rgb_top_down, cv2.COLOR_RGB2BGR))
    print(f"Captured top-down camera frame: {image_path}")

    print(f"\n[Task]: {initial_task}")

    # 2. Ambiguity Loop 
    input_data = {
        "task_description": initial_task,
        "image_path": image_path
    }
    
    result = vision_pipeline.handle_query_dict(input_data)
    
    # If the task is ambiguous, ask the user for clarification
    if result.get("task_ambiguous"):
        print(f"\n[Robot asks]: {result['clarifying_question']}")
        user_clarification = input("[Your response]: ")
        
        # Send clarification back to the VLM to get the final objects
        result = vision_pipeline.handle_response(user_clarification)
    
    final_objects = result.get("task_objects", [])
    # Ensure any items mentioned in the initial task command are preserved
    for candidate in ["red can", "blue can", "green can", "yellow cube", "purple cube"]:
        if candidate in initial_task.lower() and candidate not in final_objects:
            final_objects.append(candidate)

    print(f"\n[Grounded Objects]: {final_objects}")

    # 3. PDDL Generation & Validation
    print("\nLoading pre-defined domain and Generating Problem PDDL JSON...")
    
    # Read the predefined domain
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
    domain_path = os.path.join(repo_root, "domains", "manipulation", "domain.pddl")
    with open(domain_path, "r") as f:
        domain_pddl = f.read()

    try:
        # Calls the method designed to output JSON for the problem only
        initial_state_desc = "All objects are currently resting on the table. The robot's arm is empty."
        target_locations = ["sorting_bin", "pot"]
        problem_obj = reasoning_pipeline.generate_problem_json(
            initial_state_desc=initial_state_desc,
            task_description=initial_task, 
            task_objects=final_objects,
            domain_pddl=domain_pddl,
            target_locations=target_locations,
        )
        
        # Convert the Pydantic object to PDDL string
        problem_pddl = problem_obj.to_pddl()
        
        print("\n✅ PDDL Problem Generated Successfully!")
        
    except ValueError as e:
        print(f"\n❌ Validation Failed: {e}")
        return

    # 4. Save Trial to generated_problems Folder & Solve
    gen_problems_dir = os.path.join(repo_root, "generated_problems")
    os.makedirs(gen_problems_dir, exist_ok=True)

    # Number each trial sequentially with a timestamp
    existing_files = [f for f in os.listdir(gen_problems_dir) if f.endswith(".pddl")]
    trial_num = len(existing_files) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_filename = f"problem_trial_{trial_num:03d}_{timestamp}.pddl"
    trial_filepath = os.path.join(gen_problems_dir, trial_filename)

    with open(trial_filepath, "w") as f:
        f.write(problem_pddl)
    print(f"\n📁 Saved trial problem file to: {trial_filepath}")

    # Also keep domain.pddl and problem.pddl in root for current execution
    with open(os.path.join(repo_root, "domain.pddl"), "w") as f:
        f.write(domain_pddl)
    with open(os.path.join(repo_root, "problem.pddl"), "w") as f:
        f.write(problem_pddl)

    print("\n--- Generated Problem PDDL ---")
    print(problem_pddl)
    print("------------------------------")
        
    print("\nRunning Fast Downward Solver...")
    fast_downward_path = os.path.join(repo_root, "downward", "fast-downward.py")
    pddl_env = PDDLenv(
        fast_downward_path=fast_downward_path,
        time_limit=10,
        planning_algorithm=PDDLenv.SUB_OPTIMAL_ALIAS
    )
    
    plan, succ, msg = pddl_env.search_plan(
        domain_pddl=domain_pddl,
        problem_pddl=problem_pddl
    )
    
    if succ and plan:
        print("\n--- Fast Downward Solution Plan ---")
        print(plan)
    else:
        print(f"\n[ERROR] Failed to generate plan: {msg}")
    
    env.close()

if __name__ == "__main__":
    main()