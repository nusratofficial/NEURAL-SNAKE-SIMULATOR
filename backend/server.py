"""
FastAPI + WebSocket Backend Server for Neural Snake RL.
Runs the Snake DQN simulation in Python and streams full live telemetry
(game grid, neural activations, weight matrices, gradient deltas, performance metrics)
to the web interface.
"""
import asyncio
import json
import os
import math
from typing import Dict, Any, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.snake_env import SnakeGame
from backend.dqn_agent import DQNAgent

app = FastAPI(title="Neural Snake Python Backend")

# Initialize Game & Agent
game = SnakeGame(20, 20)
agent = DQNAgent(
    layer_sizes=[12, 18, 12, 3],
    learning_rate=0.001,
    gamma=0.95,
    epsilon=0.06,
    epsilon_decay=0.999,
    min_epsilon=0.03,
    batch_size=64
)

# Load pretrained weights if available
pretrained_path = os.path.join(os.path.dirname(__file__), "pretrained_weights.json")
if os.path.exists(pretrained_path):
    try:
        with open(pretrained_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            agent.policy_net.from_dict(data)
            agent.target_net.from_dict(data)
            print("Successfully loaded pre-trained neural weights into Python DQN agent!")
    except Exception as e:
        print(f"Warning: could not load pretrained weights: {e}")

# State control variables
is_running = True
is_manual_mode = False
manual_action = None
sim_speed = 5 # ticks per second or multiplier
high_score = 0
score_history = []
reward_weights = {
    "food": 10.0,
    "death": -15.0,
    "closer": 0.15,
    "farther": -0.2,
    "step": -0.01
}

# Connected WebSocket clients
active_connections: Set[WebSocket] = set()

def perform_tick():
    global high_score, is_running, manual_action

    state = game.get_state()
    action = 0
    is_exploration = False
    q_vals = [0.0, 0.0, 0.0]
    info_meta = {}

    if is_manual_mode:
        if manual_action is not None:
            action = manual_action
        # Forward pass on human state to compute AI evaluation
        q_vals = agent.policy_net.predict(state).tolist()
    else:
        # Agent decides with Survival Guard (Safe RL) enabled
        action, is_exploration, q_vals, info_meta = agent.act(state, game=game)

    next_state, reward, done, info = game.step(action, reward_weights)

    # Continuous Reinforcement Learning: updates policy every step!
    if not is_manual_mode:
        agent.remember(state, action, reward, next_state, done)
        agent.train_step()

    if game.score > high_score:
        high_score = game.score

    if done:
        score_history.append(game.score)
        if len(score_history) > 100:
            score_history.pop(0)

        if not is_manual_mode:
            agent.decay_epsilon()

        game.reset()

    # Build payload
    payload = {
        "type": "tick",
        "game": {
            "width": game.width,
            "height": game.height,
            "head": {"x": game.head[0], "y": game.head[1]},
            "body": [{"x": p[0], "y": p[1]} for p in game.body],
            "food": {"x": game.food[0], "y": game.food[1]},
            "direction": {"x": game.direction[0], "y": game.direction[1]},
            "score": game.score,
            "steps": game.steps,
            "isGameOver": done,
            "highScore": high_score
        },
        "agent": {
            "action": action,
            "isExploration": is_exploration,
            "qValues": q_vals,
            "loss": float(agent.latest_loss),
            "epsilon": float(agent.epsilon),
            "episodes": agent.episodes,
            "totalUpdates": agent.total_updates,
            "survivalGuard": agent.survival_guard,
            "continuousLearning": agent.continuous_learning,
            "safeActions": info_meta.get("safe_actions", [0, 1, 2]),
            "isManual": is_manual_mode
        },
        "neural": {
            "activations": agent.policy_net.last_activations,
            "layers": [
                {
                    "inputSize": l.input_size,
                    "outputSize": l.output_size,
                    "weights": l.weights.tolist(),
                    "biases": l.biases.tolist(),
                    "deltas": l.weight_deltas.tolist()
                }
                for l in agent.policy_net.layers
            ]
        }
    }
    return payload


async def broadcast_tick(payload):
    if not active_connections:
        return
    msg = json.dumps(payload)
    disconnected = set()
    for ws in active_connections:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        active_connections.remove(ws)


async def simulation_loop():
    global is_running, sim_speed
    while True:
        try:
            if is_running and active_connections:
                if sim_speed >= 100:
                    # Turbo mode: run 20 ticks per loop iteration
                    for _ in range(18):
                        perform_tick()
                    payload = perform_tick()
                    await broadcast_tick(payload)
                    await asyncio.sleep(0.016)
                else:
                    delay = max(0.01, 0.2 / max(1, sim_speed))
                    payload = perform_tick()
                    await broadcast_tick(payload)
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_running, is_manual_mode, manual_action, sim_speed, reward_weights, high_score
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action_type = cmd.get("type")

            if action_type == "pause":
                is_running = False
            elif action_type == "resume":
                is_running = True
            elif action_type == "step":
                is_running = False
                payload = perform_tick()
                await broadcast_tick(payload)
            elif action_type == "set_speed":
                sim_speed = int(cmd.get("speed", 5))
            elif action_type == "reset_game":
                game.reset()
            elif action_type == "reset_brain":
                agent.reset()
                high_score = 0
                game.reset()
            elif action_type == "load_pretrained":
                if os.path.exists(pretrained_path):
                    with open(pretrained_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        agent.policy_net.from_dict(data)
                        agent.target_net.from_dict(data)
                        agent.epsilon = 0.05
                        game.reset()
            elif action_type == "toggle_survival":
                agent.survival_guard = cmd.get("enabled", True)
            elif action_type == "toggle_continuous":
                agent.continuous_learning = cmd.get("enabled", True)
            elif action_type == "toggle_manual":
                is_manual_mode = cmd.get("manual", False)
            elif action_type == "manual_direction":
                manual_action = cmd.get("action", 0)
            elif action_type == "update_params":
                if "learningRate" in cmd:
                    agent.learning_rate = float(cmd["learningRate"])
                if "gamma" in cmd:
                    agent.gamma = float(cmd["gamma"])
                if "epsilon" in cmd:
                    agent.epsilon = float(cmd["epsilon"])
                if "foodReward" in cmd:
                    reward_weights["food"] = float(cmd["foodReward"])
                if "deathPenalty" in cmd:
                    reward_weights["death"] = float(cmd["deathPenalty"])
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)


# Serve Model Weights Download
@app.get("/api/weights")
async def get_weights():
    return JSONResponse(agent.policy_net.to_dict())

# Mount frontend web static assets
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    import webbrowser

    port = 8000
    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f"  [+] NEURAL SNAKE // PYTHON BACKEND & WEB UI ONLINE")
    print(f"  Server URL: {url}")
    print(f"=======================================================\n")
    webbrowser.open(url)
    uvicorn.run("backend.server:app", host="0.0.0.0", port=port, log_level="info")
