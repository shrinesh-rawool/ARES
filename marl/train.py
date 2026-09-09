"""
marl/train.py
Training entrypoint for MAPPO on Multi-Agent Warehouse Environment.
"""

import argparse
import os
import time
import numpy as np
from marl.environment import MultiAgentWarehouseEnv
from marl.mappo import MAPPOAgent, MAPPORolloutBuffer


def train(
    num_episodes: int = 20,
    steps_per_epoch: int = 64,
    num_agents: int = 3,
    save_dir: str = "models",
):
    print("=" * 60)
    print("🚀 Starting ARES MARL (MAPPO) Training")
    print(f"Agents: {num_agents} | Episodes: {num_episodes} | Steps/Epoch: {steps_per_epoch}")
    print("=" * 60)

    env = MultiAgentWarehouseEnv(num_agents=num_agents)
    obs_dim = env.obs_dim
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = MAPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        num_agents=num_agents,
        lr_actor=3e-4,
        lr_critic=1e-3,
        ppo_epochs=4,
        batch_size=32,
    )

    buffer = MAPPORolloutBuffer(
        num_steps=steps_per_epoch,
        num_agents=num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
    )

    os.makedirs(save_dir, exist_ok=True)
    actor_save_path = os.path.join(save_dir, "mappo_actor.pt")
    critic_save_path = os.path.join(save_dir, "mappo_critic.pt")

    start_time = time.time()
    obs_dict, state = env.reset()

    for epoch in range(1, num_episodes + 1):
        epoch_rewards = []
        epoch_collisions = 0

        for step in range(steps_per_epoch):
            # 1. Decentralized action selection
            actions, log_probs = agent.select_actions(obs_dict, deterministic=False)

            # 2. Centralized state evaluation
            value = agent.evaluate_state(state)

            # 3. Step environment
            next_obs_dict, next_state, rewards, dones, infos = env.step(actions)

            # Record stats
            step_rew = sum(rewards.values()) / num_agents
            epoch_rewards.append(step_rew)
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

        # Compute Generalized Advantage Estimation
        next_value = agent.evaluate_state(state)
        buffer.compute_gae(next_value)

        # Update Actor and Critic
        loss_info = agent.train_step(buffer)

        mean_reward = np.mean(epoch_rewards)
        print(
            f"Epoch [{epoch:03d}/{num_episodes:03d}] | "
            f"Mean Reward: {mean_reward:6.2f} | "
            f"Collisions: {epoch_collisions:2d} | "
            f"Actor Loss: {loss_info['actor_loss']:6.3f} | "
            f"Critic Loss: {loss_info['critic_loss']:6.3f}"
        )

    # Save models
    agent.save_models(actor_save_path, critic_save_path)
    # Also save to root directory for easy AMR node loading
    agent.save_models("mappo_actor.pt", "mappo_critic.pt")

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"✅ MAPPO training finished in {elapsed:.1f}s.")
    print(f"Models saved to: {actor_save_path} and {critic_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO on ARES warehouse")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=64, help="Steps per training epoch")
    parser.add_argument("--agents", type=int, default=3, help="Number of AMRs")
    parser.add_argument("--save_dir", type=str, default="models", help="Save directory")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        steps_per_epoch=args.steps,
        num_agents=args.agents,
        save_dir=args.save_dir,
    )
