# Hierarchical VLM Planning and Control with Ambiguity Resolution

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Robosuite 1.5.1](https://img.shields.io/badge/robosuite-1.5.1-orange.svg)](https://robosuite.ai/)
[![Planner Fast-Downward](https://img.shields.io/badge/planner-Fast--Downward-green.svg)](https://www.fast-downward.org/)
[![Behavior Trees py_trees](https://img.shields.io/badge/executive-py__trees-red.svg)](https://py-trees.readthedocs.io/)

A neuro-symbolic **Task and Motion Planning (TAMP)** framework that bridges high-level Vision-Language Models (VLMs) and low-level continuous robotic manipulation. The framework resolves linguistic and visual ambiguities through multi-turn dialogue, synthesizes verified PDDL problem representations, computes provably valid symbolic plans via Fast Downward, and compiles them into reactive Behavior Trees executed with collision-free $SE(3)$ Task-Space RRT motion planning on a 7-DOF Franka Emika Panda manipulator.

---

## 🏛️ System Architecture

```text
       Natural Language Instruction + Top-Down RGB Observation
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ 1. PERCEPTION & AMBIGUITY RESOLUTION (Qwen 2.5-VL 3B)             │
 │    • Zero-shot scene grounding & referential ambiguity detection  │
 │    • Interactive clarification loop (Robot ↔ Human dialogue)     │
 │    • Structured Pydantic schema extraction (Grounded Objects)     │
 └────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ 2. SYMBOLIC TASK PLANNING (LLaMA 3 + Fast Downward)               │
 │    • Automatic PDDL problem synthesis with strict type validation │
 │    • Domain validation against predefined STRIPS manipulation rules│
 │    • Heuristic state-space search (LAMA / sub-optimal alias)      │
 │    • Output: Provably sound sequential action plan π = (a₁...aₙ)   │
 └────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ 3. EXECUTIVE LAYER (py_trees Behavior Tree Synthesis)             │
 │    • Dynamic compilation of PDDL plan into hierarchical BT        │
 │    • Sequence memory, pre-/post-condition guards, and retry loops │
 │    • State monitoring with active skill / stage introspection     │
 └────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │ 4. CONTINUOUS MOTION PLANNING & CONTROL (TaskSpaceRRT)            │
 │    • 3D Cartesian obstacle avoidance (bounding box collision check)│
 │    • Greedy trajectory shortcutting and waypoint interpolation     │
 │    • Staged primitives: Pre-grasp ➔ Approach ➔ Grasp ➔ Lift ➔ Place│
 │    • Robosuite (MuJoCo) execution with live dual-camera OpenCV HUD │
 └───────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Scientific Contributions

1. **Grounded Interactive Ambiguity Resolution:** Instead of executing hallucinations or failing under ambiguous commands (e.g. *"put the can in the bin"* with multiple cans), the VLM identifies semantic under-specification against scene affordances and generates targeted queries to ground exact entity references.
2. **Formal Neuro-Symbolic Translation Guardrails:** Generated PDDL definitions are strictly validated via Pydantic schemas enforcing typing, variable binding, and operator consistency before entering the heuristic solver, eliminating syntax-level solver crashes.
3. **Reactive Plan Execution via Behavior Trees:** Converts static linear PDDL plans into reactive execution graphs capable of monitoring task progress, recovering from minor perturbations, and coordinating hybrid primitives (sampling-based RRT, Imitation Learning BC, and RL policies).
4. **Collision-Free Cartesian Path Synthesis:** Combines semantic task dispatching with real-time `TaskSpaceRRT` collision avoidance over complex workspace geometries (sorting bins, pots, and dynamic object obstacles).

---

## 📂 Repository Structure

```text
vlm_pddl_tamp/
├── src/
│   ├── ambiguityres/        # VLM inference (Qwen 2.5-VL), prompt engineering, schema validation
│   ├── llm2pddl/            # LLM PDDL problem generator, Pydantic type validator, Fast Downward interface
│   ├── behaviorTree/        # py_trees BT builder, condition nodes, TaskSpaceRRT motion planner, skills
│   └── envs/                # Robosuite 'TaskSorting' environment, SpacedSamplers, CustomArena XML
├── domains/
│   └── manipulation/        # STRIPS PDDL domain definition (pick, place, fixtures)
├── downward/                # Fast Downward heuristic search planner
├── tests/
│   ├── test_vlm_to_pddl_simulation.py  # Primary end-to-end simulation benchmark (VLM -> BT -> RRT)
│   ├── test_ambres_pddl.py             # Ambiguity resolution & PDDL translation verification
│   ├── test_vlm_to_pddl.py             # Standalone VLM-to-PDDL compilation test
│   └── test_env.py                     # Robosuite environment visual verification
├── generated_problems/      # Automated log of generated PDDL trial problems
└── videos/                  # Multi-camera simulation trial recordings (.mp4)
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
# Activate conda environment
conda activate robomimic_venv

# Ensure local packages are linked in compatibility editable mode
pip install -e ./robosuite -e ./robomimic --no-deps --config-settings editable_mode=compat
```

### 2. End-to-End Simulation (VLM + PDDL + BT + RRT)

Run the full interactive pipeline with live OpenCV dual-camera GUI (Frontal + Top-Down VLM view):

```bash
python tests/test_vlm_to_pddl_simulation.py \
  --task "Put the red can in the sorting bin, then put the yellow cube in the pot. Then take the blue can and put it in the sorting bin."
```

### 3. Rapid Benchmark / Headless Mode (Deterministic Plan)

Bypass the external VLM server to benchmark the symbolic planner, Behavior Tree compiler, and Cartesian RRT motion planner:

```bash
# Headless run without GUI (generates video in videos/)
python tests/test_vlm_to_pddl_simulation.py --skip-vlm --no-gui --max-ticks 1200
```

### 4. Modular Unit Tests

- **Environment & Cameras:** `python tests/test_env.py`
- **Ambiguity & PDDL Generation:** `python tests/test_vlm_to_pddl.py`
- **Ambiguity Loop with Behavior Tree:** `python tests/test_ambres_pddl.py`

---

## 📊 Experimental Setup

- **Manipulator:** 7-DOF Franka Emika Panda with 2-finger parallel gripper.
- **Simulation Engine:** MuJoCo / Robosuite 1.5.1 operating at 20 Hz control frequency.
- **Vision Sensors:**
  - `top_down_vlm`: Orthographic overhead camera ($512 \times 512$) for scene grounding and state estimation.
  - `frontview`: Oblique perspective camera ($512 \times 512$) for trajectory tracking and HUD visualization.
- **Symbolic Solver:** Fast Downward with `lama-first` heuristic search engine.
- **Motion Planning:** 3D Cartesian TaskSpaceRRT with goal bias ($20\%$), step size $\delta = 0.05\,\text{m}$, bounding-box collision avoidance, and path shortcutting.
