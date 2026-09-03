"""
train_rl.py — Headless Reinforcement Learning Training Script for AMR Fleet.

Trains a Deep Q-Network (DQN) policy to navigate the 30x30 warehouse,
avoiding static shelves, dynamic obstacle walls, and peers solely through
reward/penalty reinforcement signals without hardcoded rules.
"""

import argparse
import os
import sys
import time
from typing import List

import numpy as np

from core.grid import WarehouseGrid
from core.layouts import LAYOUT_NAMES
from core.rl_agent import DQNAgent
from core.rl_env import AMRWarehouseEnv


def train(
    episodes: int = 1000,
    batch_size: int = 64,
    target_update_interval: int = 10,
    save_path: str = "models/amr_rl_policy.pt",
    num_agents: int = 4,
) -> DQNAgent:
    print("=" * 60)
    print(f"  AMR Fleet MVP — Multi-Agent RL Training ({num_agents} AMRs)")
    print(f"  Episodes: {episodes} | Batch size: {batch_size} | Model: {save_path}")
    print(f"  Scenarios ({len(LAYOUT_NAMES)}): {', '.join(LAYOUT_NAMES)}")
    print("=" * 60)

    grid = WarehouseGrid()
    env = AMRWarehouseEnv(grid=grid, max_steps=120, num_agents=num_agents)
    agent = DQNAgent(device="cpu")

    # If existing model exists, warm-start
    if os.path.exists(save_path):
        print(f"[Trainer] Loading pre-existing model from {save_path} to continue training…")
        agent.load(save_path, epsilon=0.25)

    recent_rewards: List[float] = []
    recent_fleet_successes: List[bool] = []
    recent_collisions: List[int] = []

    start_time = time.time()

    for ep in range(1, episodes + 1):
        # Cycle through layouts every 3 episodes, with random obstacles
        layout_idx = (ep // 3) % len(LAYOUT_NAMES)
        obs_dict = env.reset(randomize_obstacles=True, layout=layout_idx)

        total_reward = 0.0
        collisions = 0
        dones = {aid: False for aid in env.agent_ids}

        while not all(dones.values()): # type: ignore
            actions_dict: Dict[str, int] = {}
            for aid in env.agent_ids:
                if not dones[aid]:
                    actions_dict[aid] = agent.select_action(obs_dict[aid])
                else:
                    actions_dict[aid] = 4

            next_obs_dict, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)

            # Push experiences from all agents into the shared replay memory
            for aid in env.agent_ids:
                agent.memory.push(
                    obs_dict[aid],
                    actions_dict[aid],
                    rewards_dict[aid],
                    next_obs_dict[aid],
                    dones_dict[aid],
                )
                total_reward += rewards_dict[aid]
                if infos_dict[aid]["collision"]:
                    collisions += 1

            agent.update(batch_size=batch_size)
            obs_dict = next_obs_dict
            dones = dones_dict

        # Update target network
        if ep % target_update_interval == 0:
            agent.update_target_network()

        fleet_success = all(infos_dict[aid]["success"] for aid in env.agent_ids)
        recent_rewards.append(total_reward / num_agents)
        recent_fleet_successes.append(fleet_success)
        recent_collisions.append(collisions)

        # Logging
        if ep % 25 == 0 or ep == episodes:
            avg_reward = np.mean(recent_rewards[-25:])
            fleet_win_rate = np.mean(recent_fleet_successes[-25:]) * 100.0
            avg_coll = np.mean(recent_collisions[-25:])
            elapsed = time.time() - start_time
            print(
                f"[Ep {ep:4d}/{episodes}]  Avg Reward: {avg_reward:6.1f} | "
                f"Fleet Win: {fleet_win_rate:5.1f}% | Collisions: {avg_coll:4.1f} | "
                f"Eps: {agent.epsilon:0.3f} | {elapsed:.1f}s"
            )

    # Save trained policy
    agent.save(save_path)
    print(f"\n[Trainer] Training complete. Policy saved to {save_path}")

    # Evaluation run
    print("\n[Trainer] Evaluating trained policy on 4-agent fleet scenario:")
    eval_obs = env.reset(randomize_obstacles=True, layout=0)
    eval_dones = {aid: False for aid in env.agent_ids}
    steps = 0
    while not all(eval_dones.values()) and steps < 120:
        acts = {aid: agent.select_action(eval_obs[aid], eval_mode=True) for aid in env.agent_ids}
        eval_obs, _, eval_dones, eval_infos = env.step(acts)
        steps += 1

    success_count = sum(1 for aid in env.agent_ids if eval_infos[aid]["success"])
    print(f"[Trainer] Fleet Evaluation: {success_count}/{len(env.agent_ids)} AMRs reached goals in {steps} steps.")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMR RL Policy")
    parser.add_argument("--episodes",   type=int, default=600, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=64,  help="Batch size for experience replay")
    parser.add_argument("--num-agents", type=int, default=4,   help="Number of AMRs in training fleet (1-4)")
    parser.add_argument("--save-path",  type=str, default="models/amr_rl_policy.pt", help="Path to save model")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        batch_size=args.batch_size,
        save_path=args.save_path,
        num_agents=args.num_agents,
    )

