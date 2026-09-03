# Reinforcement Learning (DQN) Autonomous Navigation for AMR Fleet

## Overview
We replaced hardcoded collision conditions and heuristic path search with an **independent Reinforcement Learning (Deep Q-Network)** policy. Each AMR observes its surroundings through local spatial sensors and a goal direction vector, choosing its actions entirely through a neural network policy trained via reward/penalty reinforcement signals.

---

## 1. System Architecture

```
                                  AMR Node (Local Sensor)
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      ▼                                                                             ▼
Goal Vector (dx, dy, dist)                                        Local 5×5 Perception Window
      │                                                                             │
      └──────────────────────────────────────┬──────────────────────────────────────┘
                                             ▼
                              28-Dimensional State Observation
                                             │
                                             ▼
                             Deep Q-Network (128 ➔ 64 ➔ 5)
                                             │
                                             ▼
                        Learned Action: [N, S, W, E, Wait]
```

### Observation Space (28-Dimensional Vector)
- **Goal Direction**: Normalized vector `(gx - x)/30.0` and `(gy - y)/30.0`.
- **Manhattan Distance**: Normalized distance to target `dist / 60.0`.
- **5×5 Local Spatial Sensor**: Centered on the robot, detecting static shelves, boundary walls, dynamic obstacles, and peer AMRs (`0.0` = free, `1.0` = blocked).

### Action Space (5 Discrete Actions)
- `0: North (0, -1)`
- `1: South (0, 1)`
- `2: West (-1, 0)`
- `3: East (1, 0)`
- `4: Wait (0, 0)`

### Reward Function $R(s, a)$
- **Goal Reached**: `+100.0`
- **Obstacle / Wall / Peer Collision**: `-25.0`
- **Step Cost**: `-0.4` (encourages shortest paths)
- **Distance Shaping**: `+2.5 × (prev_dist - curr_dist)`

---

## 2. Key Components Built

1. [core/rl_env.py]
   - Fast, headless multi-agent warehouse simulation environment.
   - Vectorized observation builder and dynamic obstacle randomizer for generalized avoidance training.
2. [core/rl_agent.py]
   - `QNetwork`: PyTorch neural network with 2 hidden layers (128 and 64 units).
   - `ReplayBuffer`: Experience replay memory of size 25,000.
   - `DQNAgent`: Independent Q-learning with target network stabilization and $\epsilon$-greedy exploration decay.
3. [train_rl.py]
   - Curriculum training script training the agent over benchmark warehouse corridors and randomized obstacle layouts.
   - Saves trained weights to `models/amr_rl_policy.pt`.
4. [node.py]
   - Supports `--policy rl` (default) and `--policy planner`.
   - In RL mode, evaluates sensory inputs directly with the trained neural network to take movements and projects a rollout for telemetry visualization.
5. [run_fleet.py]
   - Added `--policy` argument passing the desired mode (`rl` by default) to all launched AMR edge processes.

---

## 3. Training & Evaluation Results

Training completed across 500 episodes in **36.3 seconds**:

```
[Ep   50/500]  Reward:   75.1 | Success:  70.0% | Collisions: 1.1 | Eps: 0.050 | 5.1s
[Ep  100/500]  Reward:  125.1 | Success:  84.0% | Collisions: 0.2 | Eps: 0.050 | 9.1s
[Ep  250/500]  Reward:  142.3 | Success:  96.0% | Collisions: 0.2 | Eps: 0.050 | 18.9s
[Ep  450/500]  Reward:  144.3 | Success:  96.0% | Collisions: 0.0 | Eps: 0.050 | 32.6s
[Ep  500/500]  Reward:  138.2 | Success:  90.0% | Collisions: 0.0 | Eps: 0.050 | 36.3s

[Trainer] Evaluating trained policy on test scenarios:
  Test route 1 ((2, 14) -> (27, 14)): REACHED GOAL in 25 steps (Collisions: 0)
  Test route 2 ((14, 2) -> (14, 27)): REACHED GOAL in 25 steps (Collisions: 0)
  Test route 3 ((2, 27) -> (27, 2)): REACHED GOAL in 50 steps (Collisions: 0)
[Trainer] Evaluation success rate: 3/3 (100%)
```

### Real-Time Obstacle Reaction Test
When an obstacle wall was dynamically placed in front of AMR 1 at `(11, 14), (12, 14), (13, 14)`:
- Sensor front cell shifted to `1.0` (blocked).
- RL Policy immediately selected action `0` (North to `(10, 13)`), steering completely clear of the wall along row 13 without any predefined detour rule or A* tree search!

---

## 4. Running the System

### Launch Fleet with Trained RL Policy (Default)
```powershell
python run_fleet.py --policy rl
```

### (Optional) Compare with Classical Space-Time Planner
```powershell
python run_fleet.py --policy planner
```

### Retrain Policy (with more episodes if desired)
```powershell
python train_rl.py --episodes 800
```
