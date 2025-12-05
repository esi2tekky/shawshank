"""
src/attacker/rl_policy.py
-------------------------
Policy and Value networks for RL-based adversarial prompt optimization.

Implements Actor-Critic architecture for learning to select effective
(intent, strategy) combinations for jailbreaking target models.

Key Components:
- State encoder: Converts (intent, history, context) to feature vector
- Policy network (Actor): π(action | state) - selects strategy
- Value network (Critic): V(state) - estimates expected attack success
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import taxonomy
from src.attacker.bandit_attacker import (
    INTENT_CATEGORIES,
    STRATEGIES,
    generate_prompt,
    get_intent_action,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_INTENTS = len(INTENT_CATEGORIES)  # 11
NUM_STRATEGIES = len(STRATEGIES)       # 11
NUM_STRATEGY_COMBOS = NUM_STRATEGIES + (NUM_STRATEGIES * (NUM_STRATEGIES - 1)) // 2  # 11 + 55 = 66

INTENT_LIST = list(INTENT_CATEGORIES.keys())
STRATEGY_LIST = list(STRATEGIES.keys())

# Build action space (single strategies + 2-combinations)
ACTION_SPACE: List[Tuple[str, ...]] = []
for s in STRATEGY_LIST:
    ACTION_SPACE.append((s,))
for i, s1 in enumerate(STRATEGY_LIST):
    for s2 in STRATEGY_LIST[i+1:]:
        ACTION_SPACE.append((s1, s2))

NUM_ACTIONS = len(ACTION_SPACE)


# ============================================================================
# STATE REPRESENTATION
# ============================================================================

@dataclass
class RLState:
    """
    State representation for RL agent.

    Contains all information the agent uses to select actions.
    """
    intent_idx: int                    # Index into INTENT_LIST
    step: int = 0                      # Current step in episode (0-indexed)
    prev_action_idx: int = -1          # Previous action taken (-1 if first step)
    prev_reward: float = 0.0           # Reward from previous step
    cumulative_reward: float = 0.0     # Total reward so far in episode

    # Response features (from previous attempt)
    response_length: int = 0           # Length of model response
    refusal_detected: bool = False     # Whether model refused

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        features = []

        # Intent one-hot (11 dims)
        intent_onehot = [0.0] * NUM_INTENTS
        intent_onehot[self.intent_idx] = 1.0
        features.extend(intent_onehot)

        # Step number (normalized, 1 dim)
        features.append(self.step / 5.0)  # Normalize assuming max 5 steps

        # Previous action one-hot (66 dims) - all zeros if first step
        prev_action_onehot = [0.0] * NUM_ACTIONS
        if self.prev_action_idx >= 0:
            prev_action_onehot[self.prev_action_idx] = 1.0
        features.extend(prev_action_onehot)

        # Previous reward (1 dim)
        features.append(self.prev_reward)

        # Cumulative reward (1 dim)
        features.append(self.cumulative_reward)

        # Response features (2 dims)
        features.append(min(self.response_length / 500.0, 2.0))  # Normalized
        features.append(1.0 if self.refusal_detected else 0.0)

        return torch.tensor(features, dtype=torch.float32)

    @staticmethod
    def state_dim() -> int:
        """Return dimensionality of state vector."""
        return NUM_INTENTS + 1 + NUM_ACTIONS + 1 + 1 + 2  # 11 + 1 + 66 + 1 + 1 + 2 = 82


# ============================================================================
# POLICY NETWORK (ACTOR)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Actor network that outputs action probabilities given state.

    Architecture: State -> FC -> ReLU -> FC -> ReLU -> Action logits
    """

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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for layer in [self.fc1, self.fc2, self.action_head]:
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action logits.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Action logits of shape (batch_size, num_actions) or (num_actions,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)
        return logits

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor
            deterministic: If True, return argmax action

        Returns:
            Tuple of (action_idx, log_prob)
        """
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
        """Get log probability of taking action in state."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(torch.tensor(action))


# ============================================================================
# VALUE NETWORK (CRITIC)
# ============================================================================

class ValueNetwork(nn.Module):
    """
    Critic network that estimates state value V(s).

    Predicts expected cumulative reward from current state.
    """

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
        """
        Forward pass returning value estimate.

        Args:
            state: State tensor

        Returns:
            Value estimate (scalar per batch element)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value.squeeze(-1)


# ============================================================================
# ACTOR-CRITIC COMBINED
# ============================================================================

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network with shared feature extraction.

    More parameter-efficient than separate networks.
    """

    def __init__(
        self,
        state_dim: int = RLState.state_dim(),
        hidden_dim: int = 128,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim // 2, num_actions)

        # Critic head (value)
        self.critic = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and value.

        Args:
            state: State tensor

        Returns:
            Tuple of (action_logits, value)
        """
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value estimate.

        Args:
            state: State tensor
            deterministic: If True, return argmax action

        Returns:
            Tuple of (action_idx, log_prob, value)
        """
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
        """
        Evaluate log prob, value, and entropy for given state-action pair.

        Used during training for computing losses.

        Args:
            state: Batch of states
            action: Batch of actions

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, value, entropy


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def action_idx_to_strategies(action_idx: int) -> List[str]:
    """Convert action index to list of strategy names."""
    if action_idx < 0 or action_idx >= len(ACTION_SPACE):
        raise ValueError(f"Invalid action index: {action_idx}")
    return list(ACTION_SPACE[action_idx])


def strategies_to_action_idx(strategies: List[str]) -> int:
    """Convert list of strategy names to action index."""
    strategies_tuple = tuple(sorted(strategies))
    for idx, action in enumerate(ACTION_SPACE):
        if tuple(sorted(action)) == strategies_tuple:
            return idx
    raise ValueError(f"Unknown strategy combination: {strategies}")


def intent_to_idx(intent: str) -> int:
    """Convert intent name to index."""
    return INTENT_LIST.index(intent)


def idx_to_intent(idx: int) -> str:
    """Convert index to intent name."""
    return INTENT_LIST[idx]


def detect_refusal(response: str) -> bool:
    """Simple heuristic to detect if model refused."""
    refusal_phrases = [
        "i can't", "i cannot", "i won't", "i will not",
        "i'm not able", "i am not able",
        "against my", "not appropriate",
        "harmful", "dangerous", "illegal",
        "i apologize", "sorry",
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in refusal_phrases)


# ============================================================================
# TESTING
# ============================================================================

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
