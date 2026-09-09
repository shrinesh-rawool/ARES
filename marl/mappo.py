"""
marl/mappo.py
Multi-Agent Proximal Policy Optimization (MAPPO) with CTDE.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from marl.models import ActorNetwork, CriticNetwork
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MAPPORolloutBuffer:
    """Stores experience tuples across timesteps for centralized training."""

    def __init__(self, num_steps: int, num_agents: int, obs_dim: int, state_dim: int):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        self.obs = np.zeros((num_steps, num_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((num_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_agents), dtype=np.int64)
        self.log_probs = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((num_steps,), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_agents), dtype=np.float32)

        self.step = 0

    def insert(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        value: float,
    ):
        self.obs[self.step] = obs
        self.states[self.step] = state
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = value
        self.step += 1

    def compute_gae(self, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation across all agents."""
        # Mean team reward per timestep for centralized value comparison
        mean_rewards = np.mean(self.rewards, axis=1)

        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_val = next_value
                next_done = 0.0
            else:
                next_val = self.values[t + 1]
                next_done = np.max(self.dones[t])

            delta = mean_rewards[t] + gamma * next_val * (1.0 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae

            for i in range(self.num_agents):
                # Individual agent advantage combines team baseline with agent-specific reward delta
                agent_delta = self.rewards[t, i] - mean_rewards[t]
                self.advantages[t, i] = last_gae + agent_delta
                self.returns[t, i] = self.advantages[t, i] + self.values[t]

        # Reset pointer
        self.step = 0


class MAPPOAgent:
    """
    Centralized Training, Decentralized Execution MAPPO Agent.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 5,
        state_dim: int = 111,
        num_agents: int = 3,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MAPPOAgent.")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(
        self, observations: Dict[str, np.ndarray], deterministic: bool = False
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        actions = {}
        log_probs = {}

        self.actor.eval()
        with torch.no_grad():
            for agent_id, obs in observations.items():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                if deterministic:
                    logits = self.actor(obs_t)
                    act = torch.argmax(logits, dim=-1).item()
                    log_p = 0.0
                else:
                    act_t, log_p_t, _ = self.actor.get_action_and_logprob(obs_t)
                    act = act_t.item()
                    log_p = log_p_t.item()
                actions[agent_id] = act
                log_probs[agent_id] = log_p

        return actions, log_probs

    def evaluate_state(self, state: np.ndarray) -> float:
        self.critic.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            val = self.critic(state_t).item()
        return val

    def train_step(self, buffer: MAPPORolloutBuffer) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()

        # Flatten experience
        num_steps = buffer.num_steps
        num_agents = buffer.num_agents

        flat_obs = buffer.obs.reshape(-1, self.obs_dim)
        flat_actions = buffer.actions.reshape(-1)
        flat_old_log_probs = buffer.log_probs.reshape(-1)
        flat_advantages = buffer.advantages.reshape(-1)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # State data for critic
        states = buffer.states
        returns = np.mean(buffer.returns, axis=1)

        # Convert to torch tensors
        t_obs = torch.tensor(flat_obs, dtype=torch.float32, device=self.device)
        t_actions = torch.tensor(flat_actions, dtype=torch.int64, device=self.device)
        t_old_log_probs = torch.tensor(flat_old_log_probs, dtype=torch.float32, device=self.device)
        t_advantages = torch.tensor(flat_advantages, dtype=torch.float32, device=self.device)

        t_states = torch.tensor(states, dtype=torch.float32, device=self.device)
        t_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        updates = 0

        dataset_size = t_obs.size(0)

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                b_obs = t_obs[batch_idx]
                b_actions = t_actions[batch_idx]
                b_old_log_probs = t_old_log_probs[batch_idx]
                b_advantages = t_advantages[batch_idx]

                # Actor Loss
                _, curr_log_probs, entropy = self.actor.get_action_and_logprob(b_obs, b_actions)
                ratio = torch.exp(curr_log_probs - b_old_log_probs)

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_entropy += entropy.mean().item()
                updates += 1

            # Critic Loss Update
            critic_indices = np.random.permutation(num_steps)
            for c_start in range(0, num_steps, max(1, self.batch_size // num_agents)):
                c_idx = critic_indices[c_start : c_start + max(1, self.batch_size // num_agents)]
                b_states = t_states[c_idx]
                b_returns = t_returns[c_idx]

                values = self.critic(b_states)
                critic_loss = nn.functional.mse_loss(values, b_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_critic_loss += critic_loss.item()

        return {
            "actor_loss": total_actor_loss / max(1, updates),
            "critic_loss": total_critic_loss / max(1, updates),
            "entropy": total_entropy / max(1, updates),
        }

    def save_models(self, actor_path: str = "mappo_actor.pt", critic_path: str = "mappo_critic.pt"):
        os.makedirs(os.path.dirname(os.path.abspath(actor_path)), exist_ok=True)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"[MAPPO] Saved actor to {actor_path} and critic to {critic_path}")

    def load_models(self, actor_path: str = "mappo_actor.pt", critic_path: Optional[str] = None):
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            print(f"[MAPPO] Loaded actor weights from {actor_path}")
        if critic_path and os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            print(f"[MAPPO] Loaded critic weights from {critic_path}")
