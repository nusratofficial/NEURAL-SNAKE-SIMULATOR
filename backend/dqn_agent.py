"""
Double Deep Q-Network (DDQN) Agent with Experience Replay,
Safe Action Masking (Survival Guard), and Continuous Perpetual Online Learning.
"""
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from backend.neural_net import NeuralNetwork

class ReplayBuffer:
    def __init__(self, capacity: int = 40000):
        self.capacity = capacity
        self.buffer = []
        self.pointer = 0

    def push(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool):
        item = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pointer] = item
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.pointer = 0


class DQNAgent:
    def __init__(self, layer_sizes: List[int] = None, learning_rate: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 0.08, epsilon_decay: float = 0.999,
                 min_epsilon: float = 0.03, batch_size: int = 64, tau: float = 0.01):
        if layer_sizes is None:
            layer_sizes = [12, 18, 12, 3]

        self.layer_sizes = layer_sizes
        self.policy_net = NeuralNetwork(layer_sizes)
        self.target_net = self.policy_net.clone()
        self.memory = ReplayBuffer(40000)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.tau = tau

        # Learning & Survival state flags
        self.survival_guard = True      # Guarantees survival and avoids suicidal moves
        self.continuous_learning = True  # Never stops learning, keeps updating weights
        self.steps = 0
        self.episodes = 0
        self.total_updates = 0
        self.latest_loss = 0.0
        self.latest_q_values = [0.0, 0.0, 0.0]
        self.is_training = True

    def act(self, state: List[float], game=None, force_exploit: bool = False) -> Tuple[int, bool, List[float], Dict[str, Any]]:
        """
        Action selection with Neural Q-values + Safe Action Masking & Survival Instincts.
        """
        q_vals = self.policy_net.predict(state)
        self.latest_q_values = q_vals.tolist()
        num_actions = self.layer_sizes[-1]

        # Space & Safety Analysis
        analyses = []
        if game is not None and self.survival_guard:
            for a in range(num_actions):
                res = game.analyze_action_space(a)
                analyses.append(res)
        else:
            # Neutral analysis if game not provided
            analyses = [{"safe": True, "collision": False, "space": 400, "can_reach_tail": True, "is_trap": False} for _ in range(num_actions)]

        safe_actions = [a for a in range(num_actions) if analyses[a]["safe"] and not analyses[a]["collision"]]
        non_collision_actions = [a for a in range(num_actions) if not analyses[a]["collision"]]

        # Fallback candidates if completely boxed in
        candidate_actions = safe_actions if safe_actions else (non_collision_actions if non_collision_actions else list(range(num_actions)))

        # Exploration phase (Safe Exploration - explores among survivable actions)
        if not force_exploit and self.is_training and random.random() < self.epsilon:
            action = random.choice(candidate_actions)
            return action, True, self.latest_q_values, {"analyses": analyses, "safe_actions": candidate_actions}

        # Exploitation / Policy Evaluation
        if self.survival_guard and game is not None:
            # Combine Neural Q-values with Survival Space heuristics
            scored_actions = []
            for a in range(num_actions):
                an = analyses[a]
                if an["collision"]:
                    utility = -1e7
                else:
                    utility = q_vals[a]
                    # Space bonus: normalized reachable cells
                    grid_size = game.width * game.height
                    space_ratio = an["space"] / grid_size
                    utility += space_ratio * 4.0

                    # Tail reachability bonus (guaranteed survival loop)
                    if an["can_reach_tail"]:
                        utility += 8.0

                    # Trap penalty
                    if an["is_trap"]:
                        utility -= 500.0

                scored_actions.append((a, utility))

            scored_actions.sort(key=lambda x: x[1], reverse=True)
            chosen_action = scored_actions[0][0]
        else:
            # Raw Q-values
            chosen_action = int(np.argmax(q_vals))

        return chosen_action, False, self.latest_q_values, {"analyses": analyses, "safe_actions": candidate_actions}

    def remember(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool):
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1

    def train_step(self) -> float:
        """
        Perform a gradient descent step on experience replay (Double DQN with Adam).
        """
        if not self.continuous_learning:
            return self.latest_loss

        if self.memory.size() < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        training_samples = []

        next_states = np.array([exp["next_state"] for exp in batch], dtype=np.float32)
        next_policy_qs, _ = self.policy_net.forward(next_states)
        next_target_qs, _ = self.target_net.forward(next_states)

        for idx, exp in enumerate(batch):
            if exp["done"]:
                target_q = exp["reward"]
            else:
                best_action = int(np.argmax(next_policy_qs[idx]))
                target_q = exp["reward"] + self.gamma * next_target_qs[idx, best_action]

            training_samples.append({
                "state": exp["state"],
                "action": exp["action"],
                "target_q": target_q
            })

        loss = self.policy_net.train_batch(training_samples, self.learning_rate)
        self.latest_loss = loss
        self.total_updates += 1

        # Continuous Soft Update: Target net constantly adapts to policy net
        self.target_net.copy_from(self.policy_net, self.tau)
        return loss

    def decay_epsilon(self):
        self.episodes += 1
        if self.continuous_learning:
            # Cyclic/Adaptive Epsilon: never drops below min_epsilon (stays alert and keeps learning)
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            # Periodic curiosity burst every 40 episodes to discover novel strategies
            if self.episodes % 40 == 0:
                self.epsilon = min(0.15, self.epsilon + 0.05)
        else:
            if self.epsilon > self.min_epsilon:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset(self):
        self.policy_net = NeuralNetwork(self.layer_sizes)
        self.target_net = self.policy_net.clone()
        self.memory.clear()
        self.epsilon = 0.1
        self.steps = 0
        self.episodes = 0
        self.total_updates = 0
        self.latest_loss = 0.0
