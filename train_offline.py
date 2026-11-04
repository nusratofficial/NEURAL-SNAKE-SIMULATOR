"""
Offline Training Script in Python for Snake DQN Agent.
Trains the neural network using Double DQN with heuristic pre-warming,
evaluates score progression, and saves weights to backend/pretrained_weights.json.
"""
import json
import math
import argparse
from backend.snake_env import SnakeGame
from backend.dqn_agent import DQNAgent

def get_heuristic_action(game: SnakeGame) -> int:
    headings = game.get_heading_vectors()
    straight = headings["straight"]
    right = headings["right"]
    left = headings["left"]
    hx, hy = game.head
    fx, fy = game.food

    pt_straight = (hx + straight[0], hy + straight[1])
    pt_right = (hx + right[0], hy + right[1])
    pt_left = (hx + left[0], hy + left[1])

    safe_straight = not game.is_collision(pt_straight)
    safe_right = not game.is_collision(pt_right)
    safe_left = not game.is_collision(pt_left)

    candidates = []
    if safe_straight:
        candidates.append((0, math.hypot(fx - pt_straight[0], fy - pt_straight[1])))
    if safe_right:
        candidates.append((1, math.hypot(fx - pt_right[0], fy - pt_right[1])))
    if safe_left:
        candidates.append((2, math.hypot(fx - pt_left[0], fy - pt_left[1])))

    if not candidates:
        return 0
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

def main():
    parser = argparse.ArgumentParser(description="Train Snake DQN Agent in Python")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    args = parser.parse_args()

    print("==================================================")
    print("  PYTHON DEEP Q-LEARNING SNAKE TRAINING")
    print("==================================================")

    game = SnakeGame(20, 20)
    agent = DQNAgent(
        layer_sizes=[12, 18, 12, 3],
        learning_rate=0.001,
        gamma=0.95,
        batch_size=64,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.02
    )

    print("Pre-populating experience replay with 3,000 guided steps...")
    warmup_steps = 0
    while warmup_steps < 3000:
        game.reset()
        state = game.get_state()
        done = False
        while not done and warmup_steps < 3000:
            act = get_heuristic_action(game)
            next_state, reward, done, _ = game.step(act)
            agent.remember(state, act, reward, next_state, done)
            state = next_state
            warmup_steps += 1

    print(f"Replay memory warmed with {agent.memory.size()} transitions.")
    print("Pre-training policy network on memory buffer...")
    for _ in range(200):
        agent.train_step()

    print(f"Beginning active RL training for {args.episodes} episodes...\n")
    high_score = 0
    recent_scores = []

    for ep in range(1, args.episodes + 1):
        game.reset()
        state = game.get_state()
        done = False

        while not done:
            action, _, _ = agent.act(state)
            next_state, reward, done, _ = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state

        agent.decay_epsilon()
        score = game.score
        if score > high_score:
            high_score = score
        recent_scores.append(score)
        if len(recent_scores) > 50:
            recent_scores.pop(0)

        avg_score = sum(recent_scores) / len(recent_scores)

        if ep % 25 == 0 or ep == args.episodes:
            print(
                f"Episode {ep:3d}/{args.episodes} | Score: {score:2d} | High: {high_score:2d} | "
                f"Avg(50): {avg_score:5.2f} | Epsilon: {agent.epsilon:5.3f} | Loss: {agent.latest_loss:6.4f}"
            )

    # Save trained weights
    output_path = "backend/pretrained_weights.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agent.policy_net.to_dict(), f, indent=2)

    print(f"\nTraining completed! Pretrained weights saved to {output_path}")

if __name__ == "__main__":
    main()

