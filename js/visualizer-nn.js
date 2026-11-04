/**
 * Neural Network Canvas Visualizer
 * Renders nodes, dynamic synapses, activation glows, signal pulse animations, and interactive tooltips.
 */

export class NeuralVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.signalPulses = [];
    this.hoverInfo = null;

    this.resize();
    window.addEventListener("resize", () => this.resize());

    // Mouse tracking for tooltips
    this.canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
    this.canvas.addEventListener("mouseleave", () => {
      this.hoverInfo = null;
    });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;

    this.canvas.width = (rect.width || 680) * dpr;
    this.canvas.height = (rect.height || 440) * dpr;
    this.ctx.scale(dpr, dpr);
    this.displayW = rect.width || 680;
    this.displayH = rect.height || 440;
  }

  onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (!this.nodeLayout) {
      this.hoverInfo = null;
      return;
    }

    // Check node hover
    for (let l = 0; l < this.nodeLayout.length; l++) {
      const layerNodes = this.nodeLayout[l];
      for (let i = 0; i < layerNodes.length; i++) {
        const node = layerNodes[i];
        const dist = Math.hypot(node.x - mouseX, node.y - mouseY);
        if (dist <= node.radius + 4) {
          this.hoverInfo = {
            type: "node",
            layerIndex: l,
            nodeIndex: i,
            x: mouseX,
            y: mouseY,
            node
          };
          return;
        }
      }
    }

    // Check synapse hover (within line threshold)
    if (this.currentNetwork) {
      for (let l = 0; l < this.currentNetwork.layers.length; l++) {
        const layer = this.currentNetwork.layers[l];
        const fromNodes = this.nodeLayout[l];
        const toNodes = this.nodeLayout[l + 1];

        for (let i = 0; i < layer.outputSize; i++) {
          const toNode = toNodes[i];
          for (let j = 0; j < layer.inputSize; j++) {
            const fromNode = fromNodes[j];
            const dist = this.distToSegment(
              mouseX,
              mouseY,
              fromNode.x,
              fromNode.y,
              toNode.x,
              toNode.y
            );
            if (dist < 4.0) {
              this.hoverInfo = {
                type: "synapse",
                layerIndex: l,
                from: j,
                to: i,
                weight: layer.weights[i][j],
                delta: layer.weightDeltas[i][j],
                x: mouseX,
                y: mouseY
              };
              return;
            }
          }
        }
      }
    }

    this.hoverInfo = null;
  }

  distToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  spawnPulse(fromX, fromY, toX, toY, color = "#00f3ff", speed = 0.04) {
    if (this.signalPulses.length > 90) return; // Limit particles for performance
    this.signalPulses.push({
      fromX,
      fromY,
      toX,
      toY,
      progress: 0,
      speed: speed + Math.random() * 0.02,
      color
    });
  }

  updatePulses() {
    for (let i = this.signalPulses.length - 1; i >= 0; i--) {
      const p = this.signalPulses[i];
      p.progress += p.speed;
      if (p.progress >= 1.0) {
        this.signalPulses.splice(i, 1);
      }
    }
  }

  render(network, featureNames = [], actionNames = [], chosenAction = -1, isTraining = true) {
    this.currentNetwork = network;
    const ctx = this.ctx;
    const w = this.displayW;
    const h = this.displayH;

    // Clear background
    ctx.fillStyle = "#0c0e17";
    ctx.fillRect(0, 0, w, h);

    if (!network || !network.layerSizes) return;

    const layerSizes = network.layerSizes;
    const numLayers = layerSizes.length;

    // Compute layout
    const marginX = 85;
    const marginY = 32;
    const usableW = w - marginX * 2;
    const usableH = h - marginY * 2;
    const layerSpacingX = usableW / (numLayers - 1);

    const layout = [];

    for (let l = 0; l < numLayers; l++) {
      const count = layerSizes[l];
      const x = marginX + l * layerSpacingX;
      const nodesInLayer = [];

      // Determine radius based on node count
      const radius = count > 14 ? 7 : count > 8 ? 9 : 13;
      const spacingY = usableH / (count + 1);

      for (let i = 0; i < count; i++) {
        const y = marginY + (i + 1) * spacingY;
        let act = 0;
        if (network.lastActivations && network.lastActivations[l]) {
          act = network.lastActivations[l][i] || 0;
        }

        let bias = 0;
        if (l > 0 && network.layers[l - 1]) {
          bias = network.layers[l - 1].biases[i] || 0;
        }

        nodesInLayer.push({ x, y, radius, activation: act, bias });
      }
      layout.push(nodesInLayer);
    }
    this.nodeLayout = layout;

    // 1. Draw Synapses
    for (let l = 0; l < network.layers.length; l++) {
      const layer = network.layers[l];
      const fromLayer = layout[l];
      const toLayer = layout[l + 1];

      for (let i = 0; i < layer.outputSize; i++) {
        const toNode = toLayer[i];
        for (let j = 0; j < layer.inputSize; j++) {
          const fromNode = fromLayer[j];
          const weight = layer.weights[i][j];
          const absW = Math.abs(weight);

          if (absW < 0.03) continue; // Skip negligible weights for clarity

          const isPos = weight > 0;
          const alpha = Math.min(0.85, Math.max(0.12, absW * 0.45));
          const lineWidth = Math.min(3.2, Math.max(0.8, absW * 1.4));

          ctx.strokeStyle = isPos
            ? `rgba(0, 243, 255, ${alpha})`
            : `rgba(255, 0, 127, ${alpha})`;
          ctx.lineWidth = lineWidth;

          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);
          ctx.stroke();

          // Occasionally spawn pulse if connection is strongly active
          if (
            Math.random() < 0.015 &&
            Math.abs(fromNode.activation * weight) > 0.4
          ) {
            this.spawnPulse(
              fromNode.x,
              fromNode.y,
              toNode.x,
              toNode.y,
              isPos ? "#00f3ff" : "#ff007f"
            );
          }
        }
      }
    }

    // 2. Draw Synaptic Signal Pulses
    this.updatePulses();
    for (const p of this.signalPulses) {
      const px = p.fromX + (p.toX - p.fromX) * p.progress;
      const py = p.fromY + (p.toY - p.fromY) * p.progress;

      ctx.save();
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 3. Draw Neurons
    for (let l = 0; l < numLayers; l++) {
      const nodes = layout[l];
      const isInput = l === 0;
      const isOutput = l === numLayers - 1;

      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const act = node.activation;
        const normAct = Math.max(0, Math.min(1, isOutput ? (act + 5) / 10 : act));

        ctx.save();

        // Node Glow based on activation
        if (normAct > 0.05) {
          ctx.shadowColor = isOutput
            ? chosenAction === i
              ? "#ffe600"
              : "#00f3ff"
            : "#00f3ff";
          ctx.shadowBlur = Math.min(18, normAct * 20);
        }

        // Inner fill
        const rColor = Math.round(18 + normAct * 200);
        const gColor = Math.round(22 + normAct * 230);
        const bColor = Math.round(38 + normAct * 255);
        ctx.fillStyle = `rgb(${rColor}, ${gColor}, ${bColor})`;

        // Outer border
        ctx.strokeStyle =
          isOutput && chosenAction === i
            ? "#ffe600"
            : normAct > 0.3
            ? "#00f3ff"
            : "#3a4768";
        ctx.lineWidth = isOutput && chosenAction === i ? 2.8 : 1.5;

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // If chosen winning action, draw pulsing outer indicator
        if (isOutput && chosenAction === i) {
          ctx.strokeStyle = "rgba(255, 230, 0, 0.7)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 4, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.restore();

        // Labels for Input Layer
        if (isInput && featureNames[i]) {
          ctx.save();
          ctx.fillStyle = normAct > 0.4 ? "#00f3ff" : "#8a99ad";
          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          // Abbreviate for clean layout
          const label = featureNames[i]
            .replace("Danger ", "Dang.")
            .replace("Distance to Food", "Dist.Food");
          ctx.fillText(label, node.x - node.radius - 6, node.y);
          ctx.restore();
        }

        // Labels and Q-values for Output Layer
        if (isOutput && actionNames[i]) {
          ctx.save();
          const isChosen = chosenAction === i;
          ctx.fillStyle = isChosen ? "#ffe600" : "#8a99ad";
          ctx.font = isChosen
            ? "bold 11px 'JetBrains Mono', monospace"
            : "10px 'JetBrains Mono', monospace";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";

          const actionLabel = actionNames[i];
          const qVal = act.toFixed(2);
          ctx.fillText(`${actionLabel} [Q: ${qVal}]`, node.x + node.radius + 8, node.y);
          ctx.restore();
        }
      }

      // Layer Header Title
      const layerTitles = ["INPUTS (12)", "HIDDEN 1 (18)", "HIDDEN 2 (12)", "ACTIONS (3)"];
      ctx.save();
      ctx.fillStyle = "#5c6b84";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(layerTitles[l] || `L${l}`, nodes[0].x, marginY - 14);
      ctx.restore();
    }

    // 4. Render Tooltip if hovered
    if (this.hoverInfo) {
      this.renderTooltip(ctx);
    }
  }

  renderTooltip(ctx) {
    const info = this.hoverInfo;
    ctx.save();

    let title = "";
    let lines = [];

    if (info.type === "node") {
      const layerNames = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"];
      title = `${layerNames[info.layerIndex]} [Node ${info.nodeIndex}]`;
      lines.push(`Activation: ${info.node.activation.toFixed(4)}`);
      if (info.layerIndex > 0) {
        lines.push(`Bias: ${info.node.bias.toFixed(4)}`);
      }
    } else if (info.type === "synapse") {
      title = `Synapse L${info.layerIndex} [${info.from} -> ${info.to}]`;
      lines.push(`Weight: ${info.weight.toFixed(4)}`);
      lines.push(`Last Delta: ${info.delta ? info.delta.toExponential(3) : "0.00"}`);
    }

    ctx.font = "11px 'JetBrains Mono', monospace";
    let maxW = ctx.measureText(title).width;
    for (const l of lines) {
      maxW = Math.max(maxW, ctx.measureText(l).width);
    }

    const boxW = maxW + 20;
    const boxH = (lines.length + 1) * 16 + 14;
    let bx = info.x + 12;
    let by = info.y + 12;
    if (bx + boxW > this.displayW) bx = info.x - boxW - 8;
    if (by + boxH > this.displayH) by = info.y - boxH - 8;

    ctx.fillStyle = "rgba(16, 20, 32, 0.94)";
    ctx.strokeStyle = "#00f3ff";
    ctx.lineWidth = 1;
    ctx.shadowColor = "rgba(0, 243, 255, 0.5)";
    ctx.shadowBlur = 10;

    ctx.fillRect(bx, by, boxW, boxH);
    ctx.strokeRect(bx, by, boxW, boxH);

    ctx.shadowBlur = 0;
    ctx.fillStyle = "#00f3ff";
    ctx.textAlign = "left";
    ctx.fillText(title, bx + 10, by + 16);

    ctx.fillStyle = "#e0e6ed";
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], bx + 10, by + 32 + i * 16);
    }

    ctx.restore();
  }
}

