"""
marl/train_curriculum.py
Curriculum Learning Pipeline for Multi-Agent Reinforcement Learning (MAPPO).
Progressively trains from 2 AMRs -> 3 AMRs -> 4 AMRs -> 5 AMRs with Domain Randomization & Dynamic Obstacles.
"""

import argparse
import os
import time
import numpy as np
from marl.environment import MultiAgentWarehouseEnv
from marl.mappo import MAPPOAgent, MAPPORolloutBuffer


def run_curriculum_training(
    stages: list = None,
    steps_per_epoch: int = 64,
    save_dir: str = "models",
):
    if stages is None:
        stages = [
            {"stage": 1, "agents": 2, "episodes": 60, "obstacles": 0.0, "random_tasks": False, "desc": "1-on-1 Head-on & Corridor Clearance"},
            {"stage": 2, "agents": 3, "episodes": 80, "obstacles": 0.04, "random_tasks": True, "desc": "3-AMR Choke Point & Task Randomization"},
            {"stage": 3, "agents": 4, "episodes": 80, "obstacles": 0.08, "random_tasks": True, "desc": "4-AMR Congestion & Dynamic Obstacles"},
            {"stage": 4, "agents": 5, "episodes": 80, "obstacles": 0.10, "random_tasks": True, "desc": "5-AMR High-Density Fleet Mastery"},
        ]

    print("=" * 75)
    print("🎓 ARES MARL CURRICULUM LEARNING PIPELINE")
    print("Progressive Scaling: 2 -> 3 -> 4 -> 5 AMRs | Domain Randomization | Obstacles")
    print("=" * 75)

    # Base environment to inspect dimensions
    base_env = MultiAgentWarehouseEnv(num_agents=3)
    obs_dim = base_env.obs_dim      # 45
    state_dim = base_env.state_dim  # 225
    action_dim = base_env.action_dim

    # Persistent Agent across curriculum stages (preserves learned policy)
    agent = MAPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        num_agents=5,
        lr_actor=3e-4,
        lr_critic=1e-3,
        ppo_epochs=4,
        batch_size=32,
    )

    os.makedirs(save_dir, exist_ok=True)
    start_total_time = time.time()

    for config in stages:
        stage_num = config["stage"]
        n_agents = config["agents"]
        n_episodes = config["episodes"]
        obs_prob = config["obstacles"]
        random_tasks = config["random_tasks"]
        desc = config["desc"]

        print("\n" + "-" * 75)
        print(f"📍 STAGE {stage_num}/4: {n_agents} AMRs | {n_episodes} Episodes | Obstacle Prob: {obs_prob:.2f}")
        print(f"   Objective: {desc}")
        print("-" * 75)

        env = MultiAgentWarehouseEnv(
            num_agents=n_agents,
            max_steps=120,
            randomize_tasks=random_tasks,
            dynamic_obstacle_prob=obs_prob,
        )

        buffer = MAPPORolloutBuffer(
            num_steps=steps_per_epoch,
            num_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
        )

        obs_dict, state = env.reset()

        for epoch in range(1, n_episodes + 1):
            epoch_rewards = []
            epoch_collisions = 0
            deliveries = 0

            for step in range(steps_per_epoch):
                actions, log_probs = agent.select_actions(obs_dict, deterministic=False)
                value = agent.evaluate_state(state)

                next_obs_dict, next_state, rewards, dones, infos = env.step(actions)

                step_rew = sum(rewards.values()) / n_agents
                epoch_rewards.append(step_rew)

                for a_id, r in rewards.items():
                    if r >= 20.0:
                        deliveries += 1

                for info in infos.values():
                    if info.get("collision", False):
                        epoch_collisions += 1

                # Convert to arrays for buffer
                obs_arr = np.stack([obs_dict[a_id] for a_id in env.agent_ids])
                actions_arr = np.array([actions[a_id] for a_id in env.agent_ids], dtype=np.int64)
                log_probs_arr = np.array([log_probs[a_id] for a_id in env.agent_ids], dtype=np.float32)
                rewards_arr = np.array([rewards[a_id] for a_id in env.agent_ids], dtype=np.float32)
                dones_arr = np.array([float(dones[a_id]) for a_id in env.agent_ids], dtype=np.float32)

                buffer.insert(
                    obs=obs_arr,
                    state=state,
                    actions=actions_arr,
                    log_probs=log_probs_arr,
                    rewards=rewards_arr,
                    dones=dones_arr,
                    value=value,
                )

                obs_dict = next_obs_dict
                state = next_state

                if dones["__all__"]:
                    obs_dict, state = env.reset()

            next_value = agent.evaluate_state(state)
            buffer.compute_gae(next_value)
            loss_info = agent.train_step(buffer)

            mean_reward = np.mean(epoch_rewards)
            if epoch % 10 == 0 or epoch == n_episodes or epoch == 1:
                print(
                    f"   [Stage {stage_num} | Ep {epoch:03d}/{n_episodes:03d}] "
                    f"Reward: {mean_reward:6.2f} | "
                    f"Collisions: {epoch_collisions:2d} | "
                    f"Deliveries: {deliveries:2d} | "
                    f"Actor Loss: {loss_info['actor_loss']:6.3f} | "
                    f"Critic Loss: {loss_info['critic_loss']:7.2f}"
                )

        # Stage Checkpoint
        stage_ckpt = os.path.join(save_dir, f"mappo_actor_stage_{stage_num}.pt")
        agent.save_models(stage_ckpt, os.path.join(save_dir, f"mappo_critic_stage_{stage_num}.pt"))

    # Save final generalized weights
    final_actor = os.path.join(save_dir, "mappo_actor.pt")
    final_critic = os.path.join(save_dir, "mappo_critic.pt")
    agent.save_models(final_actor, final_critic)
    agent.save_models("mappo_actor.pt", "mappo_critic.pt")

    total_elapsed = time.time() - start_total_time
    print("=" * 75)
    print(f"🎉 Curriculum Learning Complete in {total_elapsed:.1f}s!")
    print(f"Final generalized policy saved to: {final_actor} and mappo_actor.pt")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARES MARL Curriculum Learning")
    parser.add_argument("--steps", type=int, default=64, help="Steps per training epoch")
    parser.add_argument("--save_dir", type=str, default="models", help="Save directory")
    args = parser.parse_args()

    run_curriculum_training(steps_per_epoch=args.steps, save_dir=args.save_dir)
