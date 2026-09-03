# Multi-Scenario Layouts & Live Training Visualizer

## Overview
We will expand the AMR Reinforcement Learning system so agents learn across **diverse warehouse layouts and obstacle scenarios** (rather than a single fixed map). Additionally, we will build a dedicated **interactive Pygame training visualizer** (`visualize_training.py`) where you can visually watch the AMR explore, learn, avoid obstacles, and improve its Q-values in real time.

---

## 1. Multi-Layout Warehouse Generator (`core/layouts.py`)

Create 6 distinct warehouse environments to prevent overfitting and guarantee generalization:

1. **Scenario 1: Standard Dual-Corridor** — Traditional warehouse with two shelf rows leaving center corridors at $x=14, y=14$.
2. **Scenario 2: Dense Storage Aisles** — Multiple narrow parallel aisles (testing navigation in tight corridors).
3. **Scenario 3: Central Chokepoints / Tunnels** — Wall dividing the warehouse with two narrow doorways (testing bottleneck traversal).
4. **Scenario 4: Pillar / Column Field** — Regular grid of 2 × 2 pillars throughout the open floor (testing obstacle weaving).
5. **Scenario 5: Asymmetric Distribution Center** — Diagonal / L-shaped rack clusters with loading dock zones.
6. **Scenario 6: Dynamic Clutter & Debris** — Open floor with randomly spawned pallet clusters and changing barriers.

Each episode during training randomly selects a layout and randomizes start/goal pairs and dynamic obstacle walls.

---

## 2. Interactive Live Training Visualizer (`visualize_training.py`)

A standalone Pygame training dashboard that renders the learning process live on screen:

### Visual Display
- **Warehouse Grid**: Render the active layout (shelves, dynamic obstacles, borders, goal marker).
- **Robot Body & Sensor Cone**:
  - Highlights the AMR and its 5 × 5 local sensory field.
  - Leaves a fading motion trail showing the path explored in the current episode.
- **Side Panel & Live Metrics**:
  - **Episode & Layout**: Current episode number and layout name.
  - **Live Q-Value Bar Chart**: Real-time bar graph displaying the neural network's Q-values for all 5 actions (`North`, `South`, `West`, `East`, `Wait`), showing how confident the model is in each direction.
  - **Exploration Rate (ϵ)**: Shows the transition from random exploration to learned policy exploitation.
  - **Episode Reward & Steps**: Total reward accumulated in the current episode.
  - **Success Rate & Collisions**: Running 50-episode win rate % and collision frequency.

### Interactive Controls
- **`SPACE`**: Pause / Resume training.
- **`F`**: Toggle **Fast-Forward mode** (runs at max computation speed without drawing every frame for quick training, or normal 30 FPS for visual inspection).
- **`N`**: Instantly switch to the next warehouse layout.
- **`S`**: Save current model weights to `models/amr_rl_policy.pt`.
- **`ESC` / `Q`**: Clean exit with automated model save.

---

## 3. Environment & Trainer Enhancements

- **`core/rl_env.py`**:
  - Add `set_layout(layout_name)` or `load_layout(obstacles_set)`.
  - Dynamically swap the static obstacles and rebuild the observation sensor per episode.
- **`train_rl.py`**:
  - Support multi-scenario training in headless CLI mode as well.

---

## Verification Plan

1. **Layout Integrity Test**: Verify all 6 layouts generate valid traversable paths between arbitrary points.
2. **Environment Multi-Scenario Test**: Verify `env.reset()` properly swaps layouts without memory leaks.
3. **Live Visualizer Smoke Test**: Launch `python visualize_training.py`, verify HUD, Q-value bars, and episode progression.
4. **Fast-Forward & Save Test**: Test pressing `F` to speed up training, verify model checkpoint saved on exit.
