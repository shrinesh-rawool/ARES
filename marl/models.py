"""
marl/models.py
Neural network architectures for Decentralized Actor and Centralized Critic (CTDE).
"""

from typing import Optional, Tuple
try:
    import torch
    import torch.nn as nn
    from torch.distributions.categorical import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = object


if HAS_TORCH:
    class ActorNetwork(nn.Module):
        """
        Decentralized Actor Network:
        Maps agent local observation o_i -> categorical action distribution over {Advance, Yield, Sidestep, Halt, Creep}.
        """
        def __init__(self, obs_dim: int, action_dim: int = 5, hidden_dim: int = 128):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.network(obs)

        def get_action_and_logprob(
            self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = self.forward(obs)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, entropy


    class CriticNetwork(nn.Module):
        """
        Centralized Critic Network:
        Maps joint observation S = [o_1, o_2, ..., o_N] -> scalar state value V(S).
        Used strictly during centralized training (CTDE).
        """
        def __init__(self, state_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.network(state).squeeze(-1)

else:
    # Placeholder classes when torch is not yet imported
    class ActorNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ActorNetwork.")

    class CriticNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for CriticNetwork.")
