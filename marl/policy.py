"""
marl/policy.py
Inference policy for edge AMR nodes running MAPPO.
Supports PyTorch Actor model inference with fallback to rule-based conflict arbitration.
"""

import os
from typing import Optional
import numpy as np

try:
    import torch
    from marl.models import ActorNetwork
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MAPPOPolicy:
    """
    Edge inference policy deployed on AMR nodes.
    Evaluates local observation o_i -> discrete action {ADVANCE, YIELD, SIDESTEP, HALT, CREEP}.
    """

    ACTION_ADVANCE = 0
    ACTION_YIELD = 1
    ACTION_SIDESTEP = 2
    ACTION_HALT = 3
    ACTION_CREEP = 4

    def __init__(
        self,
        obs_dim: int = 45,
        action_dim: int = 5,
        model_path: Optional[str] = "mappo_actor.pt",
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device_str = device
        self.actor = None
        self.has_weights = False

        if HAS_TORCH:
            self.device = torch.device(device)
            self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
            if model_path and os.path.exists(model_path):
                try:
                    self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.actor.eval()
                    self.has_weights = True
                    print(f"[MAPPOPolicy] Successfully loaded weights from {model_path}")
                except Exception as e:
                    print(f"[MAPPOPolicy] Warning: Could not load weights from {model_path}: {e}")
                    self.has_weights = False
        else:
            print("[MAPPOPolicy] PyTorch not detected. Running in heuristic fallback mode.")

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        Takes local observation o_i and returns action integer in [0, 4].
        """
        if HAS_TORCH and self.actor is not None and self.has_weights:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                if deterministic:
                    logits = self.actor(obs_t)
                    return int(torch.argmax(logits, dim=-1).item())
                else:
                    act_t, _, _ = self.actor.get_action_and_logprob(obs_t)
                    return int(act_t.item())

        # Fallback / Expert baseline arbitration:
        # Obs layout:
        # [0:25] = 5x5 occupancy grid
        # [25:29] = self state: dx_goal, dy_goal, dist_goal, payload
        # [29:33] = peer 1: p_dx, p_dy, p_norm_dist, has_priority
        # [33:37] = peer 2: p_dx, p_dy, p_norm_dist, has_priority
        if len(obs) >= 33:
            p1_dx, p1_dy, _, p1_priority = obs[29:33]
            peer_manhattan = abs(p1_dx * 30.0) + abs(p1_dy * 30.0)

            # If peer is within 2 cells and we don't have priority, yield or sidestep
            if peer_manhattan <= 2.0 and p1_priority < 0.5:
                # If peer is directly in front, sidestep if possible, else yield
                return self.ACTION_YIELD

        # By default, advance along planned path
        return self.ACTION_ADVANCE
