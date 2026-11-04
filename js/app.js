/**
 * Main Application Controller for Neural Snake Dashboard
 * Ties together Game, DQN Agent, Neural Visualizer, Matrix Inspector, Audio, and UI.
 */

import { SnakeGame } from "./snake.js";
import { DQNAgent } from "./dqn.js";
import { GameVisualizer } from "./visualizer-game.js";
import { NeuralVisualizer } from "./visualizer-nn.js";
import { MatrixVisualizer } from "./visualizer-matrix.js";
import { CyberAudio } from "./audio.js";
import { PRETRAINED_WEIGHTS } from "./pretrained.js";

class App {
  constructor() {
    this.initElements();
    this.initInstances();
    this.initChart();
    this.bindEvents();

    this.isRunning = true;
    this.isManualMode = false;
    this.manualDirection = null;

    this.simSpeed = 5; // default 5 ticks per sec, up to 100+
    this.lastTickTime = 0;
    this.frameCount = 0;
    this.fps = 60;
    this.lastFpsUpdate = performance.now();

    // Stats history for graphs
    this.scoreHistory = [];
    this.avgScoreHistory = [];
    this.lossHistory = [];
    this.epsilonHistory = [];
    this.highScore = 0;

    // Load pretrained weights by default for immediate enjoyment
    if (PRETRAINED_WEIGHTS) {
      try {
        this.agent.policyNet.fromJSON(PRETRAINED_WEIGHTS);
        this.agent.targetNet.fromJSON(PRETRAINED_WEIGHTS);
        this.agent.epsilon = 0.05; // Set to low exploration for pro performance
        this.updateParamsUI();
      } catch (e) {
        console.warn("Failed to load pretrained brain:", e);
      }
    }

    // Start render/simulation loop
    requestAnimationFrame((t) => this.loop(t));
  }

  initElements() {
    // Canvases
    this.gameCanvas = document.getElementById("gameCanvas");
    this.nnCanvas = document.getElementById("nnCanvas");
    this.matrixCanvas = document.getElementById("matrixCanvas");
    this.chartCanvas = document.getElementById("chartCanvas");

    // Header & HUD stats
    this.statEpisode = document.getElementById("statEpisode");
    this.statHighScore = document.getElementById("statHighScore");
    this.statScore = document.getElementById("statScore");
    this.statSteps = document.getElementById("statSteps");
    this.statLength = document.getElementById("statLength");
    this.statEpsilon = document.getElementById("statEpsilon");
    this.statLoss = document.getElementById("statLoss");
    this.statFps = document.getElementById("statFps");
    this.statusBadgeText = document.getElementById("statusBadgeText");

    // Action HUD
    this.hudAction = document.getElementById("hudAction");
    this.hudMode = document.getElementById("hudMode");

    // Controls
    this.btnPlayPause = document.getElementById("btnPlayPause");
    this.btnStep = document.getElementById("btnStep");
    this.btnReset = document.getElementById("btnReset");
    this.btnTurbo = document.getElementById("btnTurbo");
    this.btnManual = document.getElementById("btnManual");
    this.btnAudio = document.getElementById("btnAudio");
    this.speedSlider = document.getElementById("speedSlider");
    this.speedVal = document.getElementById("speedVal");

    // Matrix Inspector controls
    this.matrixTabs = document.querySelectorAll(".matrix-tab");
    this.matrixModes = document.querySelectorAll(".matrix-mode");

    // Model Actions
    this.btnResetBrain = document.getElementById("btnResetBrain");
    this.btnLoadPretrained = document.getElementById("btnLoadPretrained");
    this.btnExportModel = document.getElementById("btnExportModel");
    this.btnImportModel = document.getElementById("btnImportModel");
    this.fileInputModel = document.getElementById("fileInputModel");

    // Sliders
    this.sliderLR = document.getElementById("sliderLR");
    this.valLR = document.getElementById("valLR");
    this.sliderGamma = document.getElementById("sliderGamma");
    this.valGamma = document.getElementById("valGamma");
    this.sliderEpsilon = document.getElementById("sliderEpsilon");
    this.valEpsilon = document.getElementById("valEpsilon");
    this.sliderFoodR = document.getElementById("sliderFoodR");
    this.valFoodR = document.getElementById("valFoodR");
    this.sliderDeathR = document.getElementById("sliderDeathR");
    this.valDeathR = document.getElementById("valDeathR");
  }

  initInstances() {
    this.game = new SnakeGame(20, 20);
    this.agent = new DQNAgent({
      layerSizes: [12, 18, 12, 3],
      learningRate: 0.001,
      gamma: 0.95,
      epsilon: 1.0,
      epsilonDecay: 0.995,
      batchSize: 64
    });

    this.gameVis = new GameVisualizer(this.gameCanvas);
    this.nnVis = new NeuralVisualizer(this.nnCanvas);
    this.matrixVis = new MatrixVisualizer(this.matrixCanvas);
    this.audio = new CyberAudio();
  }

  initChart() {
    this.chartCtx = this.chartCanvas.getContext("2d");
    this.resizeChart();
    window.addEventListener("resize", () => this.resizeChart());
  }

  resizeChart() {
    const rect = this.chartCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.chartCanvas.width = (rect.width || 440) * dpr;
    this.chartCanvas.height = (rect.height || 220) * dpr;
    this.chartCtx.scale(dpr, dpr);
    this.chartW = rect.width || 440;
    this.chartH = rect.height || 220;
  }

  bindEvents() {
    // Play / Pause
    this.btnPlayPause.addEventListener("click", () => {
      this.isRunning = !this.isRunning;
      this.btnPlayPause.innerHTML = this.isRunning
        ? `<span>⏸</span> PAUSE`
        : `<span>▶</span> RESUME`;
      this.statusBadgeText.textContent = this.isRunning ? "RUNNING" : "PAUSED";
    });

    // Step
    this.btnStep.addEventListener("click", () => {
      this.isRunning = false;
      this.btnPlayPause.innerHTML = `<span>▶</span> RESUME`;
      this.statusBadgeText.textContent = "PAUSED";
      this.tick();
    });

    // Reset Game
    this.btnReset.addEventListener("click", () => {
      this.game.reset();
      this.updateHud();
    });

    // Speed Slider
    this.speedSlider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      this.simSpeed = val;
      this.speedVal.textContent = val >= 100 ? "MAX" : `${val}x`;
    });

    // Turbo mode toggle
    this.btnTurbo.addEventListener("click", () => {
      if (this.simSpeed < 100) {
        this.simSpeed = 100;
        this.speedSlider.value = 100;
        this.speedVal.textContent = "MAX";
        this.btnTurbo.classList.add("btn-primary");
      } else {
        this.simSpeed = 10;
        this.speedSlider.value = 10;
        this.speedVal.textContent = "10x";
        this.btnTurbo.classList.remove("btn-primary");
      }
    });

    // Audio toggle
    this.btnAudio.addEventListener("click", () => {
      const enabled = this.audio.toggle();
      this.btnAudio.textContent = enabled ? "🔊 SOUND" : "🔇 MUTED";
    });

    // Manual play toggle ("Take the wheel")
    this.btnManual.addEventListener("click", () => {
      this.isManualMode = !this.isManualMode;
      this.btnManual.classList.toggle("btn-primary", this.isManualMode);
      this.btnManual.textContent = this.isManualMode ? "🎮 AI TAKEOVER" : "🕹️ HUMAN PLAY";
      this.statusBadgeText.textContent = this.isManualMode ? "MANUAL PLAY" : "RL RUNNING";
    });

    // Keyboard controls for Manual Mode
    window.addEventListener("keydown", (e) => {
      if (!this.isManualMode) return;
      const key = e.key.toLowerCase();
      const cur = this.game.direction;

      if ((key === "arrowup" || key === "w") && cur.y !== 1) {
        this.manualDirection = { x: 0, y: -1 };
      } else if ((key === "arrowdown" || key === "s") && cur.y !== -1) {
        this.manualDirection = { x: 0, y: 1 };
      } else if ((key === "arrowleft" || key === "a") && cur.x !== 1) {
        this.manualDirection = { x: -1, y: 0 };
      } else if ((key === "arrowright" || key === "d") && cur.x !== -1) {
        this.manualDirection = { x: 1, y: 0 };
      }
    });

    // Matrix Layer selector tabs
    this.matrixTabs.forEach((tab) => {
      tab.addEventListener("click", (e) => {
        this.matrixTabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        const layerIdx = parseInt(tab.dataset.layer, 10);
        this.matrixVis.setLayer(layerIdx);
      });
    });

    // Matrix View mode buttons (weights, deltas, biases)
    this.matrixModes.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        this.matrixModes.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.matrixVis.setViewMode(btn.dataset.mode);
      });
    });

    // Model management buttons
    this.btnResetBrain.addEventListener("click", () => {
      if (confirm("Reset neural network weights and start learning from scratch?")) {
        this.agent.reset();
        this.scoreHistory = [];
        this.avgScoreHistory = [];
        this.lossHistory = [];
        this.epsilonHistory = [];
        this.highScore = 0;
        this.game.reset();
        this.updateParamsUI();
        this.updateHud();
      }
    });

    this.btnLoadPretrained.addEventListener("click", () => {
      if (PRETRAINED_WEIGHTS) {
        this.agent.policyNet.fromJSON(PRETRAINED_WEIGHTS);
        this.agent.targetNet.fromJSON(PRETRAINED_WEIGHTS);
        this.agent.epsilon = 0.03;
        this.game.reset();
        this.updateParamsUI();
        this.updateHud();
      }
    });

    this.btnExportModel.addEventListener("click", () => {
      const dataStr =
        "data:text/json;charset=utf-8," +
        encodeURIComponent(JSON.stringify(this.agent.policyNet.toJSON(), null, 2));
      const dlAnchor = document.createElement("a");
      dlAnchor.setAttribute("href", dataStr);
      dlAnchor.setAttribute("download", `snake-neural-model-${Date.now()}.json`);
      dlAnchor.click();
    });

    this.btnImportModel.addEventListener("click", () => {
      this.fileInputModel.click();
    });

    this.fileInputModel.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const json = JSON.parse(event.target.result);
          this.agent.policyNet.fromJSON(json);
          this.agent.targetNet.fromJSON(json);
          this.game.reset();
          alert("Neural weights imported successfully!");
        } catch (err) {
          alert("Error parsing weights JSON: " + err.message);
        }
      };
      reader.readAsText(file);
    });

    // Hyperparameter sliders
    this.sliderLR.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      this.agent.learningRate = val;
      this.valLR.textContent = val.toFixed(4);
    });

    this.sliderGamma.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      this.agent.gamma = val;
      this.valGamma.textContent = val.toFixed(2);
    });

    this.sliderEpsilon.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      this.agent.epsilon = val;
      this.valEpsilon.textContent = val.toFixed(2);
    });

    this.sliderFoodR.addEventListener("input", (e) => {
      this.valFoodR.textContent = e.target.value;
    });

    this.sliderDeathR.addEventListener("input", (e) => {
      this.valDeathR.textContent = e.target.value;
    });
  }

  getRewardWeights() {
    return {
      food: parseFloat(this.sliderFoodR?.value || 10),
      death: parseFloat(this.sliderDeathR?.value || -10),
      closer: 0.1,
      farther: -0.15,
      step: -0.01
    };
  }

  updateParamsUI() {
    if (this.sliderLR) {
      this.sliderLR.value = this.agent.learningRate;
      this.valLR.textContent = this.agent.learningRate.toFixed(4);
    }
    if (this.sliderGamma) {
      this.sliderGamma.value = this.agent.gamma;
      this.valGamma.textContent = this.agent.gamma.toFixed(2);
    }
    if (this.sliderEpsilon) {
      this.sliderEpsilon.value = this.agent.epsilon;
      this.valEpsilon.textContent = this.agent.epsilon.toFixed(2);
    }
  }

  tick() {
    const state = this.game.getState();
    let action = 0;
    let isExploration = false;

    if (this.isManualMode) {
      // Map manual heading vector to relative action (straight, right, left)
      if (this.manualDirection) {
        const { straight, right, left } = this.game.getHeadingVectors();
        const m = this.manualDirection;
        if (m.x === straight.x && m.y === straight.y) action = 0;
        else if (m.x === right.x && m.y === right.y) action = 1;
        else if (m.x === left.x && m.y === left.y) action = 2;
      }
      this.latestAction = action;
      // Evaluate network on human state anyway to show what AI would do
      this.agent.policyNet.forward(state);
    } else {
      // AI Decision via DQN
      const actResult = this.agent.act(state);
      action = actResult.action;
      isExploration = actResult.isExploration;
      this.latestAction = action;
    }

    const prevScore = this.game.score;
    const { state: nextState, reward, done } = this.game.step(action, this.getRewardWeights());

    // Reinforcement Learning update
    if (!this.isManualMode) {
      this.agent.remember(state, action, reward, nextState, done);
      this.agent.train();
    }

    // Audio effects & particle bursts
    if (this.game.score > prevScore) {
      const cellW = this.gameVis.displayW / this.game.width;
      const cellH = this.gameVis.displayH / this.game.height;
      this.gameVis.spawnFoodParticles(
        (this.game.head.x + 0.5) * cellW,
        (this.game.head.y + 0.5) * cellH
      );

      if (this.game.score > this.highScore) {
        this.highScore = this.game.score;
        this.audio.playHighScore();
      } else {
        this.audio.playFood();
      }
    }

    // Game Over handling
    if (done) {
      const cellW = this.gameVis.displayW / this.game.width;
      const cellH = this.gameVis.displayH / this.game.height;
      this.gameVis.spawnDeathParticles(
        (this.game.head.x + 0.5) * cellW,
        (this.game.head.y + 0.5) * cellH
      );
      this.audio.playCrash();

      // Record episode statistics
      const finalScore = this.game.score;
      this.scoreHistory.push(finalScore);
      if (this.scoreHistory.length > 80) this.scoreHistory.shift();

      const avg =
        this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
      this.avgScoreHistory.push(avg);
      if (this.avgScoreHistory.length > 80) this.avgScoreHistory.shift();

      this.lossHistory.push(this.agent.latestLoss);
      if (this.lossHistory.length > 80) this.lossHistory.shift();

      this.epsilonHistory.push(this.agent.epsilon);
      if (this.epsilonHistory.length > 80) this.epsilonHistory.shift();

      if (!this.isManualMode) {
        this.agent.decayEpsilon();
      }

      this.updateParamsUI();

      // Auto restart episode
      setTimeout(() => {
        this.game.reset();
        this.manualDirection = null;
      }, this.simSpeed >= 50 ? 5 : 200);
    }

    // Update Action readout
    const actionNames = SnakeGame.actionNames;
    this.hudAction.textContent = actionNames[action] || "STRAIGHT";
    this.hudMode.textContent = this.isManualMode
      ? "MANUAL"
      : isExploration
      ? "EXPLORE"
      : "EXPLOIT";
    this.hudMode.className = `val-mode ${
      this.isManualMode ? "explore" : isExploration ? "explore" : "exploit"
    }`;

    this.updateHud();
  }

  updateHud() {
    this.statEpisode.textContent = this.agent.episodes;
    this.statHighScore.textContent = this.highScore;
    this.statScore.textContent = this.game.score;
    this.statSteps.textContent = this.game.steps;
    this.statLength.textContent = this.game.body.length;
    this.statEpsilon.textContent = this.agent.epsilon.toFixed(3);
    this.statLoss.textContent = this.agent.latestLoss.toFixed(4);
  }

  renderPerformanceCharts() {
    const ctx = this.chartCtx;
    const w = this.chartW;
    const h = this.chartH;

    ctx.fillStyle = "#0c0e17";
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let y = 30; y < h - 20; y += 35) {
      ctx.beginPath();
      ctx.moveTo(30, y);
      ctx.lineTo(w - 10, y);
      ctx.stroke();
    }

    // Chart title and legend
    ctx.fillStyle = "#8a99ad";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("SCORE & LEARNING TRAJECTORY", 12, 18);

    ctx.fillStyle = "#00f3ff";
    ctx.fillText("■ Score", w - 180, 18);
    ctx.fillStyle = "#ffe600";
    ctx.fillText("■ Moving Avg", w - 115, 18);
    ctx.fillStyle = "#ff007f";
    ctx.fillText("■ Epsilon", w - 50, 18);

    if (this.scoreHistory.length < 2) {
      ctx.fillStyle = "#5c6b84";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for episode completion...", w / 2, h / 2);
      return;
    }

    const n = this.scoreHistory.length;
    const maxScore = Math.max(10, ...this.scoreHistory, ...this.avgScoreHistory);
    const plotLeft = 32;
    const plotRight = w - 16;
    const plotTop = 32;
    const plotBottom = h - 24;
    const plotW = plotRight - plotLeft;
    const plotH = plotBottom - plotTop;

    // Y Axis labels
    ctx.fillStyle = "#5c6b84";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText(maxScore.toFixed(0), plotLeft - 6, plotTop + 4);
    ctx.fillText("0", plotLeft - 6, plotBottom);

    // 1. Draw Raw Score Line
    ctx.strokeStyle = "rgba(0, 243, 255, 0.75)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = plotLeft + (i / (n - 1)) * plotW;
      const y = plotBottom - (this.scoreHistory[i] / maxScore) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 2. Draw Moving Average Curve
    ctx.strokeStyle = "#ffe600";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = plotLeft + (i / (n - 1)) * plotW;
      const y = plotBottom - (this.avgScoreHistory[i] / maxScore) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 3. Draw Epsilon Exploration Curve (dashed)
    ctx.save();
    ctx.strokeStyle = "rgba(255, 0, 127, 0.75)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = plotLeft + (i / (n - 1)) * plotW;
      const eps = this.epsilonHistory[i] ?? 1.0;
      const y = plotBottom - eps * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  loop(currentTime) {
    this.frameCount++;

    // Calculate FPS
    if (currentTime - this.lastFpsUpdate >= 500) {
      this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastFpsUpdate));
      this.frameCount = 0;
      this.lastFpsUpdate = currentTime;
      if (this.statFps) this.statFps.textContent = this.fps;
    }

    if (this.isRunning) {
      if (this.simSpeed >= 100) {
        // Turbo / Hyperdrive: 30 ticks per animation frame for instant training
        for (let i = 0; i < 25; i++) {
          this.tick();
        }
      } else {
        // Speed scaling: 1x = 100ms interval, 5x = 50ms interval, 20x = 10ms interval
        const stepInterval = Math.max(5, 120 / this.simSpeed);
        if (currentTime - this.lastTickTime >= stepInterval) {
          this.tick();
          this.lastTickTime = currentTime;
        }
      }
    }

    // Render components
    this.gameVis.render(this.game, currentTime);
    this.nnVis.render(
      this.agent.policyNet,
      SnakeGame.featureNames,
      SnakeGame.actionNames,
      this.latestAction ?? 0,
      this.agent.isTraining
    );
    this.matrixVis.render(this.agent.policyNet);
    this.renderPerformanceCharts();

    requestAnimationFrame((t) => this.loop(t));
  }
}

// Instantiate application on window load
window.addEventListener("DOMContentLoaded", () => {
  new App();
});

