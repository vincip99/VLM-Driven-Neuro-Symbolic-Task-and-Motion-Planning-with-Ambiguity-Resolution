"""
Test Script: Ambiguity Resolution, PDDL Plan Generation & BT Visual Execution
=============================================================================================
This script:
1. Initializes the custom Robosuite environment (`TaskSorting`).
2. Renders an observation frame from the camera (`top_down_vlm`).
3. Uses the VLM (`ambresFewShotPrompt` / `qwen2.5vl:3b`) for Ambiguity Resolution.
4. Generates a PDDL problem specification for the resolved task.
5. Invokes Fast Downward (`PDDLenv`) to compute a symbolic plan.
6. Converts the PDDL plan into an executable py_trees Behavior Tree.
7. Ticks the BT while stepping the Robosuite simulation and recording visual output.
"""

import os
import re
import cv2
import imageio
import numpy as np
from PIL import Image
import robosuite as suite
import py_trees

# Import custom environment, VLM / PDDL modules, and Behavior Tree builder
from src.envs import TaskSorting, RobosuiteEnvAdapter
from src.ambiguityres import ambresFewShotPrompt
from src.llm2pddl import Domain, PDDLenv
from src.llm2pddl.problem_domain_translation import generate_n_problem_translation_candidates
from src.behaviorTree import build_bt_from_pddl_plan


def test_ambiguity_resolution_pddl_and_bt():
    print("=" * 60)
    print("Step 1: Initializing Robosuite Environment (TaskSorting)...")
    print("=" * 60)
    
    env = suite.make(
        env_name="TaskSorting",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["top_down_vlm", "frontal_debug", "frontview"],
        camera_heights=512,
        camera_widths=512,
    )
    
    obs = env.reset()
    print("Environment reset successfully.")
    
    # ---------------------------------------------------------
    # Render and process camera frames (Top-Down and Front View)
    # ---------------------------------------------------------
    rgb_top_down = np.flipud(obs["top_down_vlm_image"])
    rgb_frontview = np.flipud(obs["frontview_image"])
    
    # Save camera views for visual verification
    top_down_filename = "test_ambres_camera_top_down.jpg"
    frontview_filename = "test_ambres_camera_frontview.jpg"
    cv2.imwrite(top_down_filename, cv2.cvtColor(rgb_top_down, cv2.COLOR_RGB2BGR))
    cv2.imwrite(frontview_filename, cv2.cvtColor(rgb_frontview, cv2.COLOR_RGB2BGR))
    print(f"Captured top-down camera frame: '{os.path.abspath(top_down_filename)}'.")
    print(f"Captured front-view camera frame: '{os.path.abspath(frontview_filename)}'.")
    
    pil_top_down = Image.fromarray(rgb_top_down)
    pil_frontview = Image.fromarray(rgb_frontview)
    
    try:
        pil_top_down.show(title="Top-Down VLM Camera View")
        pil_frontview.show(title="Front View VLM Camera View")
    except Exception as e:
        print(f"Could not open image preview windows: {e}")
    
    # Pass both camera views to VLM for enhanced multi-view recognition
    input_images = [pil_top_down, pil_frontview]
    
    # ---------------------------------------------------------
    # Step 2: Ambiguity Resolution via VLM
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Testing Ambiguity Resolution with VLM...")
    print("=" * 60)
    
    vlm = ambresFewShotPrompt()
    
    # Ambiguous instruction example: multiple cans in scene
    ambiguous_task = "Put the cube into the bin."
    print(f"\n[Task Prompt]: '{ambiguous_task}'")
    
    query_result = vlm.handle_query(task_description=ambiguous_task, image=input_images)
    print("\n--- Initial VLM Ambiguity Output ---")
    print(f"  Extracted Objects : {query_result.get('task_objects')}")
    print(f"  Is Ambiguous?     : {query_result.get('task_ambiguous')}")
    print(f"  Clarifying Q      : {query_result.get('clarifying_question')}")
    
    # Handle user clarification if task is ambiguous
    if query_result.get('task_ambiguous'):
        user_clarification = "The yellow cube."
        print(f"\n[User Clarification]: '{user_clarification}'")
        resolved_result = vlm.handle_response(response=user_clarification)
        print(f"  Resolved Objects  : {resolved_result.get('task_objects')}")
        target_obj = "yellow_cube"
    else:
        target_obj = "yellow_cube"
        
    print(f"\nFinal Grounded Target Object for PDDL Problem: {target_obj}")
    
    # ---------------------------------------------------------
    # Step 3: PDDL Generation
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Generating PDDL Problem File using VLM...")
    print("=" * 60)
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    domains_dir = os.path.abspath(os.path.join(curr_dir, "..", "domains"))
    target_domain = Domain(base_path=domains_dir, name="manipulation")
    context_domain = Domain(base_path=domains_dir, name="blocksworld")
    
    target_domain_nl = target_domain.get_domain_nl()
    target_domain_pddl = target_domain.get_domain_pddl()
    target_domain_template = target_domain.get_domain_template_pddl()
    _, _, target_problem_template = target_domain.get_problem(0)
    
    target_problem_nl = f"""
You control one robot, equipped with a gripper, capable of moving objects on a table.

Initially:
- The robot gripper is free.
- The red_can is on the table.
- The blue_can is on the table.
- The green_can is on the table.
- The yellow_cube is on the table.
- The purple_cube is on the table.

Your goal is to achieve the following configuration:
- The {target_obj} must be on the bin.
"""
    
    print("\n[Input Domain Description]:")
    print(target_domain_nl)
    print("\n[Input Problem Natural Language]:")
    print(target_problem_nl.strip())
    print("\n[Input Problem Template PDDL]:")
    print(target_problem_template)
    
    print("\nGenerating candidate problem PDDLs with validation...")
    pddl_candidates = generate_n_problem_translation_candidates(
        client=vlm,
        context_domain=context_domain,
        target_domain=target_domain,
        target_problem_nl=target_problem_nl,
        n_problems_candidates=5,
        max_tries=3
    )
    
    if not pddl_candidates:
        print("\n[ERROR] Failed to generate a valid problem PDDL.")
        env.close()
        return
    
    # ---------------------------------------------------------
    # Step 4: PDDL Plan Search via Fast Downward
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Solving PDDL Plan with Fast Downward...")
    print("=" * 60)
    
    fast_downward_path = os.path.join(curr_dir, "..", "downward", "fast-downward.py")
    pddl_env = PDDLenv(
        fast_downward_path=fast_downward_path,
        time_limit=10,
        planning_algorithm=PDDLenv.SUB_OPTIMAL_ALIAS  # "lama-first"
    )
    
    clean_pddl = None
    plan = None
    success = False
    
    print(f"\nTesting {len(pddl_candidates)} PDDL candidate(s) with Fast Downward...")
    for idx, candidate in enumerate(pddl_candidates):
        print(f"\n[Testing Candidate {idx + 1}/{len(pddl_candidates)}]")
        print("--- Problem PDDL Candidate ---")
        print(candidate)
        print("------------------------------")
        
        plan_cand, succ, msg = pddl_env.search_plan(
            domain_pddl=target_domain_pddl,
            problem_pddl=candidate
        )
        if succ and plan_cand:
            print(f"--> Candidate {idx + 1} SUCCEEDED! Solution plan found.")
            clean_pddl = candidate
            plan = plan_cand
            success = True
            break
        else:
            print(f"--> Candidate {idx + 1} failed plan search: {msg}")
    
    if success and plan:
        print("\n--- Selected Problem PDDL ---")
        print(clean_pddl)
        print("\n--- Fast Downward Solution Plan ---")
        print(plan)
        print("-----------------------------------")
    else:
        print("\n[ERROR] Failed to generate a valid plan from any PDDL candidate.")
        env.close()
        return

    # ---------------------------------------------------------
    # Step 5: Convert PDDL Plan -> Behavior Tree & Setup
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Converting PDDL Plan into Behavior Tree...")
    print("=" * 60)
    
    env_adapter = RobosuiteEnvAdapter(raw_env=env)
    
    # Ground VLM query function for GoalCheck visual condition
    def vlm_query_fn(frame, prompt: str) -> str:
        is_goal_met = env_adapter.is_in_bin(target_obj, "bin")
        return "yes" if is_goal_met else "no"

    goal_situation = f"The {target_obj} is in the bin"
    root_node = build_bt_from_pddl_plan(
        plan=plan,
        env=env_adapter,
        camera=None,
        vlm_query_fn=vlm_query_fn,
        goal_situation=goal_situation,
        num_attempts=3
    )

    bt_tree = py_trees.trees.BehaviourTree(root_node)
    bt_tree.setup(timeout=15, env=env_adapter)

    print("\n--- Generated Behavior Tree Structure ---")
    print(py_trees.display.ascii_tree(root_node))
    print("------------------------------------------")

    # ---------------------------------------------------------
    # Step 6: Ticking Behavior Tree with Camera Rendering
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Executing Behavior Tree with Camera Rendering...")
    print("=" * 60)

    max_ticks = 300
    tick_count = 0

    while tick_count < max_ticks:
        bt_tree.tick()
        status = root_node.status

        # Render visual frames from frontal and top-down cameras safely
        try:
            frame_front = env.sim.render(height=512, width=512, camera_name="frontal_debug")
        except Exception:
            frame_front = env.sim.render(height=512, width=512, camera_name="frontview")

        try:
            frame_top = env.sim.render(height=512, width=512, camera_name="top_down_vlm")
        except Exception:
            frame_top = env.sim.render(height=512, width=512, camera_name="frontview")

        frame_front_rgb = np.flipud(frame_front)
        frame_top_rgb = np.flipud(frame_top)

        tick_count += 1
        if tick_count % 25 == 0 or status != py_trees.common.Status.RUNNING:
            print(f"  [Tick {tick_count:03d}] BT Status: {status}")

        if status == py_trees.common.Status.SUCCESS:
            print(f"\n[SUCCESS] Behavior Tree completed task at tick {tick_count}!")
            break
        elif status == py_trees.common.Status.FAILURE:
            print(f"\n[FAILURE] Behavior Tree returned FAILURE at tick {tick_count}.")
            break

    env.close()
    print("Environment closed. Test finished!")


if __name__ == "__main__":
    test_ambiguity_resolution_pddl_and_bt()
