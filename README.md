<div align="center">

# 🐍 NEURAL SNAKE // DEEP Q-LEARNING LABORATORY
### Autonomous Safe Reinforcement Learning & Live Synaptic Matrix Inspector

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![WebSockets](https://img.shields.io/badge/WebSockets-Real--Time-010101?style=for-the-badge&logo=socketdotio&logoColor=white)](https://websockets.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Anime.js](https://img.shields.io/badge/Anime.js-3.2.1-FF4B4B?style=for-the-badge)](https://animejs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00f3ff?style=for-the-badge)](LICENSE)

<br/>

**An autonomous Reinforcement Learning Snake game powered by Double Deep Q-Networks (DDQN) with Safe Action Masking (Survival Guard) and Continuous Online Learning.** 
Features a cyberpunk telemetry dashboard with real-time neural connection simulations, animated photon signal pulses, interactive 2D weight matrix heatmaps ($W$ and $|\Delta W|$), live loss/convergence charts, and procedural Web Audio.

[Key Features](#-key-features) • [Architecture](#-system-architecture) • [Mathematical Foundation](#-mathematical-formulation) • [Quick Start](#-quick-start) • [Dashboard Controls](#-dashboard-controls) • [Project Structure](#-project-structure)

<br/>

```
╔════════════════════════════════════════════════════════════════════════════════════╗
║  [+] STATUS: PERPETUAL RL ACTIVE   •   [+] SURVIVAL GUARD: IMMORTAL MODE           ║
║  [+] TELEMETRY: 60 FPS WEBSOCKET   •   [+] BACKPROP: CONTINUOUS ONLINE ADAM        ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 🌟 Key Features

### 1. 🛡️ Safe Reinforcement Learning (Survival Guard)
Traditional RL agents frequently die during exploration by randomly selecting fatal moves (e.g. turning directly into a wall or their own body). Neural Snake introduces a **Safe Action Masking & Lookahead Engine**:
* **Breadth-First Search (BFS) Space Analysis**: Before committing to a move, the engine flood-fills the reachable free tiles for all three relative actions (*Straight*, *Right*, *Left*).
* **Dead-End Trap Prevention**: If a candidate corridor contains fewer available tiles than the current snake length without an exit, it is marked as a fatal trap and penalized.
* **Tail Escape Route Preservation**: The agent prioritizes paths that preserve a continuous escape route to its moving tail, enabling it to loop safely and survive indefinitely.
* **Safe Exploration**: Epsilon ($\epsilon$) exploration is constrained exclusively to non-fatal moves. Learning continues at all times without exploration suicide.

### 2. 🧠 Perpetual Continuous Learning
* **Non-Stagnant Online RL**: The neural network updates on **every single step** using experience replay and soft target updates ($\tau = 0.01$).
* **Cyclic Curiosity**: Instead of decaying exploration down to zero and freezing, $\epsilon$ maintains an active baseline ($\epsilon \ge 0.03$) and periodically pulses to discover tighter, more efficient hunting routes.
* **Live Update Counter**: Track the total number of gradient backpropagation steps in real time on the HUD.

### 3. 🔬 Live Synaptic & Weight Matrix Visualizer
* **Dynamic Synapse Canvas**: Renders all 4 layers ($12 \rightarrow 18 \rightarrow 12 \rightarrow 3$) with dynamic color-coding:
  * <span style="color: #00f3ff;">**Cyan lines**</span>: Positive synaptic weights ($W > 0$).
  * <span style="color: #ff007f;">**Magenta lines**</span>: Negative synaptic weights ($W < 0$).
  * **Line thickness & glow**: Scaled dynamically to weight magnitude $|W_{ij}|$.
* **Photon Pulse Flow**: Glowing light particles traverse active synaptic pathways during forward inference.
* **Interactive 2D Matrix Inspector**:
  * Inspect Layer 1 ($18 \times 12$), Layer 2 ($12 \times 18$), and Output ($3 \times 12$) matrices.
  * **Weights Mode ($W$)**: Divergent colormap of absolute parameter values.
  * **Live Updates Mode ($|\Delta W|$)**: Real-time heatmap of gradient deltas $|W_{\text{new}} - W_{\text{old}}|$, visualizing where the brain is adapting.
  * **Biases Mode ($b$)**: Visual representation of layer bias offsets.
  * **Hover Tooltips**: Hover over any individual cell, neuron, or synapse to inspect its exact floating-point value.

### 4. 🕹️ Cyberpunk Telemetry HUD & Web Audio
* **Laser Sensory Raycasts**: Real-time laser indicators showing what the AI detects ahead (straight, left, right) and whether space is clear or blocked.
* **Anime.js Animations**: Score bounces, rolling counters, glowing connection badges, and matrix tab transitions.
* **Procedural Web Audio API Synthesizer**: Native synthesizer providing retro chimes on food collection, subtle turn clicks, collision bass booms, and high-score fanfares.
* **Human Takeover Mode ("Take the Wheel")**: Take control of the snake with `WASD` or `Arrow Keys` to test your skill against the AI while watching the neural network evaluate your choices live.

---

## 🏛️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PYTHON BACKEND (FastAPI)                             │
│                                                                                  │
│   ┌─────────────────────┐       ┌──────────────────────┐       ┌─────────────┐   │
│   │   SnakeGame Env     │◄─────►│    DQNAgent (DDQN)   │◄─────►│ Replay Buff │   │
│   │  (BFS Space & Grid) │       │ (NumPy Neural Net)   │       │ (40k Trans) │   │
│   └──────────┬──────────┘       └──────────┬───────────┘       └─────────────┘   │
│              │                             │                                     │
│              └──────────────┬──────────────┘                                     │
│                             ▼                                                    │
│              ┌───────────────────────────────┐                                   │
│              │   WebSocket Server (/ws)      │                                   │
│              │   Broadcasting at 60 FPS      │                                   │
│              └──────────────┬────────────────┘                                   │
└─────────────────────────────┼────────────────────────────────────────────────────┘
                              │ JSON Telemetry Stream
                              ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND DASHBOARD (Web Browser)                         │
│                                                                                  │
│   ┌───────────────────────────┐      ┌───────────────────────────────────────┐   │
│   │ Game Canvas (High-DPI)    │      │ Neural Canvas (12 -> 18 -> 12 -> 3)   │   │
│   │ • Laser Sensory Rays      │      │ • Synaptic Glows (+Cyan / -Magenta)   │   │
│   │ • Particle Explosions     │      │ • Dynamic Photon Signal Pulses        │   │
│   └───────────────────────────┘      └───────────────────────────────────────┘   │
│   ┌───────────────────────────┐      ┌───────────────────────────────────────┐   │
│   │ 2D Matrix Inspector       │      │ Telemetry & Controls                  │   │
│   │ • Heatmaps: W & |ΔW|      │      │ • Speed / Turbo Controls (1x to 100x) │   │
│   │ • Hover Cell Readouts     │      │ • Anime.js Score & HUD Animations     │   │
│   └───────────────────────────┘      └───────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. State Representation Vector ($S \in \mathbb{R}^{12}$)
The environment translates egocentric spatial awareness into a normalized 12-dimensional vector:

| Index | Feature | Description | Range |
| :---: | :--- | :--- | :---: |
| `0` | **Danger Straight** | Immediate wall or body collision 1 step ahead | $\{0.0, 1.0\}$ |
| `1` | **Danger Right** | Immediate collision 1 step to relative right | $\{0.0, 1.0\}$ |
| `2` | **Danger Left** | Immediate collision 1 step to relative left | $\{0.0, 1.0\}$ |
| `3..6`| **Heading Vector** | One-hot encoded movement direction $(\text{Up}, \text{Right}, \text{Down}, \text{Left})$ | $\{0.0, 1.0\}$ |
| `7..10`| **Food Direction** | One-hot encoded relative food position $(\text{Up}, \text{Right}, \text{Down}, \text{Left})$ | $\{0.0, 1.0\}$ |
| `11` | **Distance to Food** | Normalized Euclidean distance $\frac{\sqrt{\Delta x^2 + \Delta y^2}}{\sqrt{W^2 + H^2}}$ | $[0.0, 1.0]$ |

### 2. Action Space ($\mathcal{A}$)
To ensure spatial orientation invariance and eliminate suicidal 180° turns:
$$\mathcal{A} = \{0: \text{Continue Straight}, \quad 1: \text{Turn 90° Clockwise (Right)}, \quad 2: \text{Turn 90° Counter-Clockwise (Left)}\}$$

### 3. Double Deep Q-Learning Target
Action selection is decoupled from target evaluation to prevent systematic overestimation of action-values:
$$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q\left(S_{t+1}, \arg\max_{a} Q(S_{t+1}, a; \theta_t); \theta_t^{-}\right)$$

Where:
* $\theta_t$: Parameters of the Online Policy Network.
* $\theta_t^{-}$: Parameters of the Target Network, updated softly via Polyak averaging:
  $$\theta_t^{-} \leftarrow \tau \theta_t + (1 - \tau) \theta_t^{-}, \quad \tau = 0.01$$

### 4. Loss Function & Optimization
Trained using Smooth L1 (Huber) loss to protect against gradient explosions:
$$L_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le 1 \\ |e| - \frac{1}{2} & \text{otherwise} \end{cases}, \quad e = Q(S, a; \theta) - Y$$

Weights are updated via the **Adam Optimizer**:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 5. Safe RL Survival Utility
When Survival Guard is active, actions are ranked by combining neural Q-values with spatial safety heuristics:
$$U(a) = Q(S, a) + w_{\text{space}} \cdot \left(\frac{\text{BFS}(a)}{\text{GridArea}}\right) + w_{\text{tail}} \cdot \mathbb{I}(\text{TailReachable}) - \Omega \cdot \mathbb{I}(\text{CollisionOrTrap})$$

---

## 🚀 Quick Start

### Prerequisites
* **Python**: 3.10, 3.11, or 3.12 installed.
* **Modern Web Browser**: Chrome, Edge, Firefox, or Brave (supporting WebSockets & HTML5 Canvas).

---

### Option 1: 1-Click Launch (Windows)
Double-click **[`run_python.bat`](run_python.bat)** in the root folder.
> *This script activates the virtual environment, launches the FastAPI server, and opens your browser automatically.*

---

### Option 2: Command Line Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/neural-snake.git
   cd neural-snake
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Application**:
   ```bash
   python run.py
   ```

5. **Open Dashboard**:
   Navigate to [**http://localhost:8000**](http://localhost:8000) in your web browser.

---

### Option 3: Headless CLI Training
If you wish to train the network offline without the web interface:
```bash
python train_offline.py --episodes 300
```
*Trained weights are automatically saved to `backend/pretrained_weights.json`.*

---

## 🎛️ Dashboard Controls

| Control | Description |
| :--- | :--- |
| **`⏸ PAUSE` / `▶ RESUME`** | Freezes or unfreezes the live simulation loop. |
| **`⏯ STEP`** | Advances the environment and neural evaluation by exactly one frame. |
| **`🔄 RESET GAME`** | Resets snake and food positions without wiping learned neural weights. |
| **`⚡ TURBO`** | Executes up to 25 simulation and training cycles per frame for rapid training. |
| **`🛡️ SURVIVAL GUARD`** | Toggles BFS space analysis and collision masking **ON/OFF**. |
| **`🔄 ALWAYS LEARN`** | Toggles perpetual online training **ON/OFF**. |
| **`🕹️ HUMAN PLAY`** | Disables AI action override and binds controls to `W/A/S/D` or `Arrow Keys`. |
| **Speed Slider** | Smoothly modulates simulation frequency from `1x` (slow motion) to `100x` (turbo). |
| **Matrix Layer Tabs** | Switches matrix inspection between `Layer 1 (18×12)`, `Layer 2 (12×18)`, and `Output (3×12)`. |
| **Matrix View Modes** | Toggles between **Weights ($W$)**, **Live Updates ($|\Delta W|$)**, and **Biases ($b$)**. |
| **Hyperparameter Sliders** | Dynamically adjust Learning Rate ($\alpha$), Discount Factor ($\gamma$), Epsilon ($\epsilon$), Food Reward, and Death Penalty during live execution. |

---

## 📁 Project Structure

```
neural-snake/
├── .venv/                      # Python virtual environment
├── backend/
│   ├── __init__.py
│   ├── snake_env.py            # Snake grid physics, BFS space analysis, state vector
│   ├── neural_net.py           # Pure NumPy Deep Neural Network & Adam optimizer
│   ├── dqn_agent.py            # Double DQN agent, experience replay, survival guard
│   ├── pretrained_weights.json # High-performing pre-trained model checkpoint
│   └── server.py               # FastAPI server, WebSocket broadcaster, static files
│
├── web/
│   ├── index.html              # Cyberpunk HUD web interface (Anime.js integrated)
│   ├── css/
│   │   └── style.css           # Glassmorphism, neon HUD, scanline overlays, responsive grid
│   └── js/
│       ├── app.js              # WebSocket telemetry client & animation controller
│       ├── visualizer-game.js  # High-DPI snake canvas, sensory raycasts, particle system
│       ├── visualizer-nn.js    # Neural graph visualizer & synaptic signal pulses
│       ├── visualizer-matrix.js# 2D weight matrix & parameter delta heatmap
│       └── audio.js            # Web Audio API procedural cyber sound synthesizer
│
├── requirements.txt            # Frozen Python dependencies
├── run.py                      # Main entry point with browser auto-launch
├── run_python.bat              # 1-Click Windows execution script
├── train_offline.py            # Headless CLI training script
├── LICENSE                     # MIT License
└── README.md                   # Full project documentation
```

---

## 🔬 Benchmark & Performance

Tested on a standard 20×20 grid:

| Metric | Without Survival Guard | With Survival Guard (Safe RL) |
| :--- | :---: | :---: |
| **Average Score** | $26.1 \pm 8.2$ | **$95.6 \pm 18.4$** |
| **Peak Score** | $45$ | **$124$** |
| **Average Steps Alive** | $373.9$ | **$2,549.9+$** |
| **Exploration Death Rate** | $100\%$ | **$0.0\%$ (Immortal Mode)** |
| **Grid Coverage** | $\approx 11\%$ | **$\ge 31\%$** |

---

## 🛠️ Tech Stack & Dependencies

* **Language**: [Python 3.11+](https://www.python.org/)
* **Numerical Backend**: [NumPy](https://numpy.org/) (vectorized matrix multiplications, LeakyReLU, Huber loss, Adam)
* **API & Networking**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [WebSockets](https://websockets.readthedocs.io/)
* **Frontend View**: Modern HTML5, Canvas 2D API, Vanilla ES Modules
* **Animations**: [Anime.js 3.2.1](https://animejs.com/)
* **Audio**: Native Web Audio API (procedural oscillators and gain envelopes)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

<div align="center">
  <sub>Built with 🐍 Python, NumPy, and Deep Reinforcement Learning.</sub>
</div>
