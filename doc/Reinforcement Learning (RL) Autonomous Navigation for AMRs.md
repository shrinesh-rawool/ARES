# Reinforcement Learning (RL) Autonomous Navigation for AMRs

## Overview
Replace pre-defined collision/obstacle rules and heuristic path planning with a **Reinforcement Learning (RL)** decision-making system. Each AMR will observe its local surroundings (goal direction, nearby obstacles, and peer positions) and choose its movements using a learned policy trained through reward-penalty feedback.

---

## Architecture & RL Formulation

```
┌────────────────────────────────────────────────────────┐
│                       AMR Node                         │
│                                                        │
│  [Local Observation: Goal Vector + Local Perception]   │
│                          │                             │
│                          ▼                             │
│              [Trained RL Policy / Q-Net]               │
│                          │                             │
│                          ▼                             │
│           [Action: Up, Down, Left, Right, Wait]        │
└────────────────────────────────────────────────────────┘
```

### 1. State / Observation Space
For each AMR, the observation vector includes:
- **Relative Goal Vector**: $(\Delta x, \Delta y)$ normalized direction and distance to its target goal.
- **Local Grid Sensor**: A local $5 \times 5$ grid centered on the AMR representing cell states:
  - `0`: Free space
  - `1`: Static shelf or boundary
  - `2`: Dynamic obstacle (injected wall/block)
  - `3`: Peer AMR
- **Peer Proximity**: Relative vector to the nearest peer AMR.

### 2. Action Space
Discrete 5 actions:
- `0`: Move North `(0, -1)`
- `1`: Move South `(0, 1)`
- `2`: Move West `(-1, 0)`
- `3`: Move East `(1, 0)`
- `4`: Wait `(0, 0)`

### 3. Reward Function $R(s, a)$
- **Goal Reached**: $+100$
- **Collision with Shelf, Wall, or Peer**: $-30$
- **Step Cost (Efficiency)**: $-0.5$ per step (encourages shortest route)
- **Distance Shaping**: $+2.0 \times (\text{prev\_dist} - \text{curr\_dist})$ (rewards moving toward goal, penalizes moving away)

---

## User Review Required

> [!IMPORTANT]
> **RL Training Prerequisite**:
> Reinforcement learning agents cannot navigate safely without prior training (an untrained neural network acts completely randomly and will crash into walls). 
> We will provide:
> 1. A fast **headless training script** (`train_rl.py`) that runs thousands of fast episodes in seconds to train the policy.
> 2. Pre-trained weights or quick training runner so the robots are ready to run in `run_fleet.py`.
> 
> **PyTorch vs NumPy DQN**:
> We can implement the DQN using PyTorch (`torch`), or pure `numpy` neural network. PyTorch is supported and allows clean backprop.

---

## Proposed Changes

### Component 1: RL Simulation Environment

#### [NEW] [core/rl_env.py]
- Headless multi-agent warehouse environment matching the 30×30 warehouse map.
- Generates random start/goal pairs and dynamic obstacles.
- Provides `reset()` and `step(actions)` returning observations, rewards, done flags, and collision info.

---

### Component 2: RL Agent & Policy

#### [NEW] [core/rl_agent.py]
- Deep Q-Network (DQN) architecture:
  - Input: Observation vector (Goal direction + flattened local $5 \times 5$ sensor).
  - Layers: Linear(dim, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 5).
  - Output: Q-values for the 5 actions.
- Includes Experience Replay buffer, target network, $\epsilon$-greedy exploration, and policy save/load methods (`.pt` or `.npz`).

---

### Component 3: Training Pipeline

#### [NEW] [train_rl.py]
- Headless trainer running 1,000–2,000 fast simulated episodes.
- Spawns random obstacle walls across lanes to teach agents how to detour around obstacles.
- Evaluates success rate and collision rate, saving `models/amr_policy.pt`.

---

### Component 4: Node Integration

#### [MODIFY] [node.py]
- Support `--policy rl` (with fallback to planner).
- When in RL mode:
  - Replaces `SpaceTimePlanner` with `RLAgent.act(obs)`.
  - At each tick, extracts the local $5 \times 5$ perception window and relative goal vector.
  - Executes the action directly without pre-defined collision rules.

---

## Verification Plan

### Automated Tests
1. **Environment Step Test**: Verify `core/rl_env.py` computes correct observations and collision penalties.
2. **Training Convergence Test**: Run `python train_rl.py --episodes 500` to verify reward increases and collision rate drops.
3. **Inference Test**: Run `python -c "from core.rl_agent import RLAgent; ..."` to verify action selection with obstacles.
4. **Full Fleet Smoke Test**: Run `python run_fleet.py` with the RL policy and test dynamic obstacle injection (`1`, `2`, `B`, `O`).
