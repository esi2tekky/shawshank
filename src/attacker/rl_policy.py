"""Actor-Critic RL policy for adversarial prompt optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.attacker.bandit_attacker import (
    INTENT_CATEGORIES,
    STRATEGIES,
    generate_prompt,
    get_intent_action,
)

NUM_INTENTS = len(INTENT_CATEGORIES)
NUM_STRATEGIES = len(STRATEGIES)
NUM_STRATEGY_COMBOS = NUM_STRATEGIES + (NUM_STRATEGIES * (NUM_STRATEGIES - 1)) // 2

INTENT_LIST = list(INTENT_CATEGORIES.keys())
STRATEGY_LIST = list(STRATEGIES.keys())

ACTION_SPACE: List[Tuple[str, ...]] = []
for s in STRATEGY_LIST:
    ACTION_SPACE.append((s,))
for i, s1 in enumerate(STRATEGY_LIST):
    for s2 in STRATEGY_LIST[i+1:]:
        ACTION_SPACE.append((s1, s2))

NUM_ACTIONS = len(ACTION_SPACE)


@dataclass
class RLState:
    """State representation for the RL agent."""
    intent_idx: int
    step: int = 0
    prev_action_idx: int = -1
    prev_reward: float = 0.0
    cumulative_reward: float = 0.0
    response_length: int = 0
    refusal_detected: bool = False

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for network input."""
        features = []

        intent_onehot = [0.0] * NUM_INTENTS
        intent_onehot[self.intent_idx] = 1.0
        features.extend(intent_onehot)

        features.append(self.step / 5.0)

        prev_action_onehot = [0.0] * NUM_ACTIONS
        if self.prev_action_idx >= 0:
            prev_action_onehot[self.prev_action_idx] = 1.0
        features.extend(prev_action_onehot)

        features.append(self.prev_reward)
        features.append(self.cumulative_reward)
        features.append(min(self.response_length / 500.0, 2.0))
        features.append(1.0 if self.refusal_detected else 0.0)

        return torch.tensor(features, dtype=torch.float32)

    @staticmethod
    def state_dim() -> int:
        return NUM_INTENTS + 1 + NUM_ACTIONS + 1 + 1 + 2


class PolicyNetwork(nn.Module):
    """Actor network that outputs action probabilities."""

    def __init__(
        self,
        state_dim: int = RLState.state_dim(),
        hidden_dim: int = 128,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.action_head = nn.Linear(hidden_dim // 2, num_actions)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.action_head]:
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)
        return logits

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Sample action from policy. Returns (action_idx, log_prob)."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def get_log_prob(self, state: torch.Tensor, action: int) -> torch.Tensor:
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(torch.tensor(action))


class ValueNetwork(nn.Module):
    """Critic network that estimates state value V(s)."""

    def __init__(
        self,
        state_dim: int = RLState.state_dim(),
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimate."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value.squeeze(-1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic with shared feature extraction."""

    def __init__(
        self,
        state_dim: int = RLState.state_dim(),
        hidden_dim: int = 128,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim // 2, num_actions)
        self.critic = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, value)."""
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action and get value. Returns (action_idx, log_prob, value)."""
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log prob, value, and entropy for state-action pair."""
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, value, entropy


def action_idx_to_strategies(action_idx: int) -> List[str]:
    if action_idx < 0 or action_idx >= len(ACTION_SPACE):
        raise ValueError(f"Invalid action index: {action_idx}")
    return list(ACTION_SPACE[action_idx])


def strategies_to_action_idx(strategies: List[str]) -> int:
    strategies_tuple = tuple(sorted(strategies))
    for idx, action in enumerate(ACTION_SPACE):
        if tuple(sorted(action)) == strategies_tuple:
            return idx
    raise ValueError(f"Unknown strategy combination: {strategies}")


def intent_to_idx(intent: str) -> int:
    return INTENT_LIST.index(intent)


def idx_to_intent(idx: int) -> str:
    return INTENT_LIST[idx]


def detect_refusal(response: str) -> bool:
    """Heuristic to detect if model refused."""
    refusal_phrases = [
        "i can't", "i cannot", "i won't", "i will not",
        "i'm not able", "i am not able",
        "against my", "not appropriate",
        "harmful", "dangerous", "illegal",
        "i apologize", "sorry",
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in refusal_phrases)


if __name__ == "__main__":
    print("Testing RL Policy Components")
    print("=" * 50)

    # Test state
    state = RLState(intent_idx=0, step=1, prev_reward=0.3)
    state_tensor = state.to_tensor()
    print(f"State dim: {RLState.state_dim()}")
    print(f"State tensor shape: {state_tensor.shape}")

    # Test policy network
    policy = PolicyNetwork()
    action_idx, log_prob = policy.get_action(state_tensor)
    print(f"\nPolicy action: {action_idx} -> {action_idx_to_strategies(action_idx)}")
    print(f"Log prob: {log_prob.item():.4f}")

    # Test value network
    value_net = ValueNetwork()
    value = value_net(state_tensor)
    print(f"Value estimate: {value.item():.4f}")

    # Test actor-critic
    ac = ActorCritic()
    action_idx, log_prob, value = ac.get_action_and_value(state_tensor)
    print(f"\nActor-Critic action: {action_idx} -> {action_idx_to_strategies(action_idx)}")
    print(f"Log prob: {log_prob.item():.4f}")
    print(f"Value: {value.item():.4f}")

    print("\n" + "=" * 50)
    print(f"Num intents: {NUM_INTENTS}")
    print(f"Num strategies: {NUM_STRATEGIES}")
    print(f"Num actions (strategy combos): {NUM_ACTIONS}")
    print("=" * 50)
