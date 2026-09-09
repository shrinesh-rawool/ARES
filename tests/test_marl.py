"""
tests/test_marl.py
Automated tests for Multi-Agent Reinforcement Learning (MAPPO).
"""

import os
import unittest
import numpy as np
import torch
from marl.environment import MultiAgentWarehouseEnv
from marl.models import ActorNetwork, CriticNetwork
from marl.mappo import MAPPOAgent, MAPPORolloutBuffer
from marl.policy import MAPPOPolicy


class TestMARLEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = MultiAgentWarehouseEnv(
            num_agents=3,
            max_steps=50,
            randomize_tasks=True,
            dynamic_obstacle_prob=0.2,
        )

    def test_environment_reset(self):
        obs_dict, state = self.env.reset()
        self.assertEqual(len(obs_dict), 3)
        self.assertIn("amr_1", obs_dict)
        self.assertIn("amr_2", obs_dict)
        self.assertIn("amr_3", obs_dict)

        # Dimension checks: 45 dims per agent, 225 dims for global state
        self.assertEqual(self.env.obs_dim, 45)
        self.assertEqual(self.env.state_dim, 225)
        for a_id, obs in obs_dict.items():
            self.assertEqual(obs.shape, (45,))
            self.assertFalse(np.isnan(obs).any())

        self.assertEqual(state.shape, (225,))

    def test_environment_step_and_obstacles(self):
        obs_dict, state = self.env.reset()
        actions = {
            "amr_1": MultiAgentWarehouseEnv.ACTION_ADVANCE,
            "amr_2": MultiAgentWarehouseEnv.ACTION_YIELD,
            "amr_3": MultiAgentWarehouseEnv.ACTION_SIDESTEP,
        }
        for _ in range(10):
            next_obs, next_state, rewards, dones, infos = self.env.step(actions)
            self.assertEqual(len(next_obs), 3)
            self.assertEqual(len(rewards), 3)

        self.assertIn("__all__", dones)
        self.assertFalse(dones["__all__"])

    def test_fleet_scaling(self):
        self.env.set_num_agents(5)
        obs_dict, state = self.env.reset()
        self.assertEqual(len(obs_dict), 5)
        self.assertEqual(state.shape, (225,))


class TestMAPPONetworks(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 45
        self.action_dim = 5
        self.state_dim = 225

    def test_actor_forward_and_sample(self):
        actor = ActorNetwork(self.obs_dim, self.action_dim)
        dummy_obs = torch.randn(4, self.obs_dim)
        action, log_prob, entropy = actor.get_action_and_logprob(dummy_obs)

        self.assertEqual(action.shape, (4,))
        self.assertEqual(log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        for a in action.tolist():
            self.assertIn(a, range(self.action_dim))

    def test_critic_forward(self):
        critic = CriticNetwork(self.state_dim)
        dummy_state = torch.randn(4, self.state_dim)
        value = critic(dummy_state)
        self.assertEqual(value.shape, (4,))


class TestMAPPOAgentAndBuffer(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 45
        self.action_dim = 5
        self.state_dim = 225
        self.num_agents = 5
        self.agent = MAPPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            num_agents=self.num_agents,
        )

    def test_select_actions_and_evaluate(self):
        observations = {
            f"amr_{i+1}": np.random.randn(self.obs_dim).astype(np.float32)
            for i in range(self.num_agents)
        }
        actions, log_probs = self.agent.select_actions(observations, deterministic=False)
        self.assertEqual(len(actions), 5)
        self.assertEqual(len(log_probs), 5)

        dummy_state = np.random.randn(self.state_dim).astype(np.float32)
        val = self.agent.evaluate_state(dummy_state)
        self.assertIsInstance(val, float)

    def test_buffer_gae_and_train_step(self):
        num_steps = 16
        buffer = MAPPORolloutBuffer(num_steps, self.num_agents, self.obs_dim, self.state_dim)

        for _ in range(num_steps):
            obs = np.random.randn(self.num_agents, self.obs_dim).astype(np.float32)
            state = np.random.randn(self.state_dim).astype(np.float32)
            actions = np.random.randint(0, self.action_dim, size=(self.num_agents,))
            log_probs = -np.random.rand(self.num_agents).astype(np.float32)
            rewards = np.random.randn(self.num_agents).astype(np.float32)
            dones = np.zeros(self.num_agents, dtype=np.float32)
            val = float(np.random.randn())
            buffer.insert(obs, state, actions, log_probs, rewards, dones, val)

        buffer.compute_gae(next_value=0.0)
        self.assertEqual(buffer.advantages.shape, (num_steps, self.num_agents))

        loss_info = self.agent.train_step(buffer)
        self.assertIn("actor_loss", loss_info)
        self.assertIn("critic_loss", loss_info)
        self.assertIn("entropy", loss_info)


class TestMAPPOPolicyInference(unittest.TestCase):
    def test_policy_inference(self):
        policy = MAPPOPolicy(obs_dim=45, action_dim=5, model_path="models/mappo_actor.pt")
        dummy_obs = np.random.randn(45).astype(np.float32)
        action = policy.get_action(dummy_obs, deterministic=True)
        self.assertIn(action, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
