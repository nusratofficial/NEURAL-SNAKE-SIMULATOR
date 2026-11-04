/**
 * Web Application Controller with Python WebSocket Integration & Anime.js Animations.
 * Supports Survival Guard (Safe RL) and Perpetual Continuous Learning.
 */

import { GameVisualizer } from "./visualizer-game.js";
import { NeuralVisualizer } from "./visualizer-nn.js";
import { MatrixVisualizer } from "./visualizer-matrix.js";
import { CyberAudio } from "./audio.js";

class PythonApp {
  constructor() {
    this.initElements();
    this.initInstances();
    this.initChart();
    this.bindEvents();
    this.connectWebSocket();

    this.isRunning = true;
    this.isManualMode = false;
    this.manualDirection = null;
    this.simSpeed = 5;

    this.survivalGuard = true;
    this.continuousLearning = true;

    this.scoreHistory = [];
    this.avgScoreHistory = [];
    this.lossHistory = [];
    this.epsilonHistory = [];
    this.prevScore = 0;
    this.prevEpisode = 0;

    // Feature and Action names
    this.featureNames = [
      "Danger Straight", "Danger Right", "Danger Left",
      "Dir Up", "Dir Right", "Dir Down", "Dir Left",
      "Food Up", "Food Right", "Food Down", "Food Left",
      "Distance to Food"
    ];
    this.actionNames = ["Go Straight", "Turn Right", "Turn Left"];

    // Start UI animation loop
    requestAnimationFrame((t) => this.renderLoop(t));
  }

  initElements() {
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
    this.statUpdates = document.getElementById("statUpdates");
    this.statusBadgeText = document.getElementById("statusBadgeText");
    this.statusDot = document.querySelector(".status-badge .dot");

    // Action HUD
    this.hudAction = document.getElementById("hudAction");
    this.hudMode = document.getElementById("hudMode");
    this.hudSafeActions = document.getElementById("hudSafeActions");

    // Indicators
    this.pillSurvival = document.getElementById("pillSurvival");
    this.pillLearning = document.getElementById("pillLearning");

    // Controls
    this.btnPlayPause = document.getElementById("btnPlayPause");
    this.btnStep = document.getElementById("btnStep");
    this.btnReset = document.getElementById("btnReset");
    this.btnTurbo = document.getElementById("btnTurbo");
    this.btnSurvival = document.getElementById("btnSurvival");
    this.btnContinuous = document.getElementById("btnContinuous");
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
    this.gameVis = new GameVisualizer(this.gameCanvas);
    this.nnVis = new NeuralVisualizer(this.nnCanvas);
    this.matrixVis = new MatrixVisualizer(this.matrixCanvas);
    this.audio = new CyberAudio();

    this.latestGameData = null;
    this.latestNeuralData = null;
    this.latestAgentData = null;
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

  connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const wsUrl = `${protocol}//${host}/ws`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("Connected to Python Perpetual RL Server");
      this.statusBadgeText.textContent = "RL PERPETUAL ACTIVE";
      this.statusBadgeText.style.color = "var(--neon-green)";
      if (this.statusDot) this.statusDot.style.background = "var(--neon-green)";

      if (window.anime) {
        window.anime({
          targets: "#statusBadge",
          scale: [0.95, 1.05, 1.0],
          duration: 600,
          easing: "easeOutElastic(1, .6)"
        });
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "tick") {
          this.handleTick(data);
        }
      } catch (err) {
        console.error("Error parsing WS message:", err);
      }
    };

    this.ws.onclose = () => {
      this.statusBadgeText.textContent = "RECONNECTING...";
      this.statusBadgeText.style.color = "var(--neon-pink)";
      if (this.statusDot) this.statusDot.style.background = "var(--neon-pink)";
      setTimeout(() => this.connectWebSocket(), 2000);
    };
  }

  send(cmd) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
    }
  }

  handleTick(data) {
    const game = data.game;
    const agent = data.agent;
    const neural = data.neural;

    this.latestGameData = {
      width: game.width,
      height: game.height,
      head: game.head,
      body: game.body,
      food: game.food,
      direction: game.direction,
      score: game.score,
      steps: game.steps,
      isGameOver: game.isGameOver,
      getHeadingVectors: () => {
        const dx = game.direction.x;
        const dy = game.direction.y;
        return {
          straight: { x: dx, y: dy },
          right: { x: -dy, y: dx },
          left: { x: dy, y: -dx }
        };
      },
      isCollision: (pt) => {
        if (pt.x < 0 || pt.x >= game.width || pt.y < 0 || pt.y >= game.height) return true;
        for (let i = 0; i < game.body.length - 1; i++) {
          if (game.body[i].x === pt.x && game.body[i].y === pt.y) return true;
        }
        return false;
      }
    };

    this.latestNeuralData = {
      layerSizes: [12, 18, 12, 3],
      lastActivations: neural.activations,
      layers: neural.layers.map((l) => ({
        inputSize: l.inputSize,
        outputSize: l.outputSize,
        weights: l.weights,
        biases: l.biases,
        weightDeltas: l.deltas
      }))
    };

    this.latestAgentData = agent;

    // Sound and particle animations on food collection
    if (game.score > this.prevScore) {
      const cellW = this.gameVis.displayW / game.width;
      const cellH = this.gameVis.displayH / game.height;
      this.gameVis.spawnFoodParticles(
        (game.head.x + 0.5) * cellW,
        (game.head.y + 0.5) * cellH
      );

      if (game.score > (game.highScore || 0)) {
        this.audio.playHighScore();
      } else {
        this.audio.playFood();
      }

      if (window.anime) {
        window.anime({
          targets: "#statScore",
          scale: [1, 1.4, 1],
          color: ["#ffffff", "#ffe600"],
          duration: 350,
          easing: "easeOutQuad"
        });
      }
    }

    if (game.isGameOver) {
      const cellW = this.gameVis.displayW / game.width;
      const cellH = this.gameVis.displayH / game.height;
      this.gameVis.spawnDeathParticles(
        (game.head.x + 0.5) * cellW,
        (game.head.y + 0.5) * cellH
      );
      this.audio.playCrash();

      this.scoreHistory.push(game.score);
      if (this.scoreHistory.length > 80) this.scoreHistory.shift();

      const avg = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
      this.avgScoreHistory.push(avg);
      if (this.avgScoreHistory.length > 80) this.avgScoreHistory.shift();

      this.lossHistory.push(agent.loss);
      if (this.lossHistory.length > 80) this.lossHistory.shift();

      this.epsilonHistory.push(agent.epsilon);
      if (this.epsilonHistory.length > 80) this.epsilonHistory.shift();
    }

    this.prevScore = game.score;

    // Update Decision Readout
    this.hudAction.textContent = this.actionNames[agent.action] || "STRAIGHT";
    this.hudMode.textContent = this.isManualMode
      ? "MANUAL"
      : agent.isExploration
      ? "EXPLORE (SAFE)"
      : "EXPLOIT";
    this.hudMode.className = `val-mode ${
      this.isManualMode ? "explore" : agent.isExploration ? "explore" : "exploit"
    }`;

    // Safe action letters
    const actionLetters = ["S", "R", "L"];
    const safeStr = (agent.safeActions || [0, 1, 2]).map((a) => actionLetters[a]).join(", ");
    this.hudSafeActions.textContent = `[${safeStr}]`;

    // Update HUD Stats
    this.statEpisode.textContent = agent.episodes;
    this.statHighScore.textContent = game.highScore || 0;
    this.statScore.textContent = game.score;
    this.statSteps.textContent = game.steps;
    this.statLength.textContent = game.body.length;
    this.statEpsilon.textContent = agent.epsilon.toFixed(3);
    this.statLoss.textContent = agent.loss.toFixed(4);
    this.statUpdates.textContent = agent.totalUpdates || 0;

    // Status Pills
    if (agent.survivalGuard !== undefined) {
      this.survivalGuard = agent.survivalGuard;
      this.pillSurvival.style.opacity = this.survivalGuard ? "1" : "0.35";
      this.pillSurvival.textContent = this.survivalGuard ? "🛡️ SURVIVAL GUARD ON" : "⚠️ SURVIVAL GUARD OFF";
      this.btnSurvival.classList.toggle("btn-primary", this.survivalGuard);
    }

    if (agent.continuousLearning !== undefined) {
      this.continuousLearning = agent.continuousLearning;
      this.pillLearning.style.opacity = this.continuousLearning ? "1" : "0.35";
      this.pillLearning.textContent = this.continuousLearning ? "🧠 PERPETUAL LEARNING" : "⏸️ LEARNING PAUSED";
      this.btnContinuous.classList.toggle("btn-primary", this.continuousLearning);
    }
  }

  bindEvents() {
    this.btnPlayPause.addEventListener("click", () => {
      this.isRunning = !this.isRunning;
      this.send({ type: this.isRunning ? "resume" : "pause" });
      this.btnPlayPause.innerHTML = this.isRunning
        ? `<span>⏸</span> PAUSE`
        : `<span>▶</span> RESUME`;
    });

    this.btnStep.addEventListener("click", () => {
      this.isRunning = false;
      this.btnPlayPause.innerHTML = `<span>▶</span> RESUME`;
      this.send({ type: "step" });
    });

    this.btnReset.addEventListener("click", () => {
      this.send({ type: "reset_game" });
    });

    this.speedSlider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      this.simSpeed = val;
      this.speedVal.textContent = val >= 100 ? "MAX" : `${val}x`;
      this.send({ type: "set_speed", speed: val });
    });

    this.btnTurbo.addEventListener("click", () => {
      if (this.simSpeed < 100) {
        this.simSpeed = 100;
        this.speedSlider.value = 100;
        this.speedVal.textContent = "MAX";
        this.btnTurbo.classList.add("btn-primary");
        this.send({ type: "set_speed", speed: 100 });
      } else {
        this.simSpeed = 10;
        this.speedSlider.value = 10;
        this.speedVal.textContent = "10x";
        this.btnTurbo.classList.remove("btn-primary");
        this.send({ type: "set_speed", speed: 10 });
      }
    });

    // Survival Guard Toggle (Guarantees survival)
    this.btnSurvival.addEventListener("click", () => {
      this.survivalGuard = !this.survivalGuard;
      this.send({ type: "toggle_survival", enabled: this.survivalGuard });
      this.btnSurvival.classList.toggle("btn-primary", this.survivalGuard);
      this.pillSurvival.textContent = this.survivalGuard ? "🛡️ SURVIVAL GUARD ON" : "⚠️ SURVIVAL GUARD OFF";
      this.pillSurvival.style.opacity = this.survivalGuard ? "1" : "0.35";
    });

    // Continuous Learning Toggle (Never stops learning)
    this.btnContinuous.addEventListener("click", () => {
      this.continuousLearning = !this.continuousLearning;
      this.send({ type: "toggle_continuous", enabled: this.continuousLearning });
      this.btnContinuous.classList.toggle("btn-primary", this.continuousLearning);
      this.pillLearning.textContent = this.continuousLearning ? "🧠 PERPETUAL LEARNING" : "⏸️ LEARNING PAUSED";
      this.pillLearning.style.opacity = this.continuousLearning ? "1" : "0.35";
    });

    this.btnAudio.addEventListener("click", () => {
      const enabled = this.audio.toggle();
      this.btnAudio.textContent = enabled ? "🔊 SOUND" : "🔇 MUTED";
    });

    this.btnManual.addEventListener("click", () => {
      this.isManualMode = !this.isManualMode;
      this.btnManual.classList.toggle("btn-primary", this.isManualMode);
      this.btnManual.textContent = this.isManualMode ? "🎮 AI TAKEOVER" : "🕹️ HUMAN PLAY";
      this.send({ type: "toggle_manual", manual: this.isManualMode });
    });

    window.addEventListener("keydown", (e) => {
      if (!this.isManualMode || !this.latestGameData) return;
      const key = e.key.toLowerCase();
      const headings = this.latestGameData.getHeadingVectors();
      let targetHeading = null;

      if (key === "arrowup" || key === "w") targetHeading = { x: 0, y: -1 };
      else if (key === "arrowdown" || key === "s") targetHeading = { x: 0, y: 1 };
      else if (key === "arrowleft" || key === "a") targetHeading = { x: -1, y: 0 };
      else if (key === "arrowright" || key === "d") targetHeading = { x: 1, y: 0 };

      if (targetHeading) {
        let action = 0;
        if (targetHeading.x === headings.straight.x && targetHeading.y === headings.straight.y) action = 0;
        else if (targetHeading.x === headings.right.x && targetHeading.y === headings.right.y) action = 1;
        else if (targetHeading.x === headings.left.x && targetHeading.y === headings.left.y) action = 2;
        this.send({ type: "manual_direction", action });
      }
    });

    this.matrixTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        this.matrixTabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        const layerIdx = parseInt(tab.dataset.layer, 10);
        this.matrixVis.setLayer(layerIdx);

        if (window.anime) {
          window.anime({
            targets: "#matrixCanvas",
            opacity: [0.6, 1],
            duration: 250,
            easing: "easeOutQuad"
          });
        }
      });
    });

    this.matrixModes.forEach((btn) => {
      btn.addEventListener("click", () => {
        this.matrixModes.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.matrixVis.setViewMode(btn.dataset.mode);
      });
    });

    this.btnResetBrain.addEventListener("click", () => {
      if (confirm("Reset Python neural network weights and restart learning from scratch?")) {
        this.send({ type: "reset_brain" });
        this.scoreHistory = [];
        this.avgScoreHistory = [];
      }
    });

    this.btnLoadPretrained.addEventListener("click", () => {
      this.send({ type: "load_pretrained" });
    });

    this.btnExportModel.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/weights");
        const json = await res.json();
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(json, null, 2));
        const dlAnchor = document.createElement("a");
        dlAnchor.setAttribute("href", dataStr);
        dlAnchor.setAttribute("download", `snake-python-model-${Date.now()}.json`);
        dlAnchor.click();
      } catch (err) {
        alert("Failed to export weights: " + err);
      }
    });

    const sendParamUpdate = () => {
      this.send({
        type: "update_params",
        learningRate: parseFloat(this.sliderLR.value),
        gamma: parseFloat(this.sliderGamma.value),
        epsilon: parseFloat(this.sliderEpsilon.value),
        foodReward: parseFloat(this.sliderFoodR.value),
        deathPenalty: parseFloat(this.sliderDeathR.value)
      });
    };

    this.sliderLR.addEventListener("input", (e) => {
      this.valLR.textContent = parseFloat(e.target.value).toFixed(4);
      sendParamUpdate();
    });
    this.sliderGamma.addEventListener("input", (e) => {
      this.valGamma.textContent = parseFloat(e.target.value).toFixed(2);
      sendParamUpdate();
    });
    this.sliderEpsilon.addEventListener("input", (e) => {
      this.valEpsilon.textContent = parseFloat(e.target.value).toFixed(2);
      sendParamUpdate();
    });
    this.sliderFoodR.addEventListener("input", (e) => {
      this.valFoodR.textContent = e.target.value;
      sendParamUpdate();
    });
    this.sliderDeathR.addEventListener("input", (e) => {
      this.valDeathR.textContent = e.target.value;
      sendParamUpdate();
    });
  }

  renderPerformanceCharts() {
    const ctx = this.chartCtx;
    const w = this.chartW;
    const h = this.chartH;

    ctx.fillStyle = "#0c0e17";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let y = 30; y < h - 20; y += 35) {
      ctx.beginPath();
      ctx.moveTo(30, y);
      ctx.lineTo(w - 10, y);
      ctx.stroke();
    }

    ctx.fillStyle = "#8a99ad";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("PERPETUAL RL LEARNING TRAJECTORY", 12, 18);

    ctx.fillStyle = "#00f3ff";
    ctx.fillText("■ Score", w - 180, 18);
    ctx.fillStyle = "#ffe600";
    ctx.fillText("■ Moving Avg", w - 115, 18);
    ctx.fillStyle = "#ff007f";
    ctx.fillText("■ Epsilon", w - 50, 18);

    if (this.scoreHistory.length < 2) {
      ctx.fillStyle = "#5c6b84";
      ctx.textAlign = "center";
      ctx.fillText("Perpetual RL learning in progress...", w / 2, h / 2);
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

    ctx.fillStyle = "#5c6b84";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText(maxScore.toFixed(0), plotLeft - 6, plotTop + 4);
    ctx.fillText("0", plotLeft - 6, plotBottom);

    // 1. Raw Score Line
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

    // 2. Moving Average Curve
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

    // 3. Epsilon Decay Line
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

  renderLoop(time) {
    if (this.latestGameData) {
      this.gameVis.render(this.latestGameData, time);
    }
    if (this.latestNeuralData) {
      this.nnVis.render(
        this.latestNeuralData,
        this.featureNames,
        this.actionNames,
        this.latestAgentData?.action ?? 0,
        true
      );
      this.matrixVis.render(this.latestNeuralData);
    }
    this.renderPerformanceCharts();
    requestAnimationFrame((t) => this.renderLoop(t));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new PythonApp();
});
