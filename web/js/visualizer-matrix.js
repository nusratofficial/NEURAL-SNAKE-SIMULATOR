/**
 * Interactive Weight Matrix & Parameter Inspector
 * Visualizes layer weight matrices, weight deltas, biases, and convergence statistics in real time.
 */

export class MatrixVisualizer {
  constructor(canvas, containerInfoEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.infoEl = containerInfoEl;

    this.selectedLayer = 0; // 0: L1, 1: L2, 2: L3
    this.viewMode = "weights"; // 'weights', 'deltas', 'biases'
    this.hoverCell = null;

    this.resize();
    window.addEventListener("resize", () => this.resize());

    this.canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
    this.canvas.addEventListener("mouseleave", () => {
      this.hoverCell = null;
    });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;

    this.canvas.width = (rect.width || 440) * dpr;
    this.canvas.height = (rect.height || 260) * dpr;
    this.ctx.scale(dpr, dpr);
    this.displayW = rect.width || 440;
    this.displayH = rect.height || 260;
  }

  setLayer(layerIndex) {
    this.selectedLayer = layerIndex;
  }

  setViewMode(mode) {
    this.viewMode = mode;
  }

  onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (!this.gridConfig) {
      this.hoverCell = null;
      return;
    }

    const { startX, startY, cellW, cellH, rows, cols } = this.gridConfig;
    if (
      mouseX >= startX &&
      mouseX <= startX + cols * cellW &&
      mouseY >= startY &&
      mouseY <= startY + rows * cellH
    ) {
      const col = Math.floor((mouseX - startX) / cellW);
      const row = Math.floor((mouseY - startY) / cellH);
      if (row >= 0 && row < rows && col >= 0 && col < cols) {
        this.hoverCell = { row, col, mouseX, mouseY };
        return;
      }
    }

    this.hoverCell = null;
  }

  render(network) {
    const ctx = this.ctx;
    const w = this.displayW;
    const h = this.displayH;

    // Background
    ctx.fillStyle = "#0c0e17";
    ctx.fillRect(0, 0, w, h);

    if (!network || !network.layers || !network.layers[this.selectedLayer]) {
      return;
    }

    const layer = network.layers[this.selectedLayer];
    const rows = layer.outputSize;
    const cols = layer.inputSize;

    // Header info
    ctx.fillStyle = "#8a99ad";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    const modeLabel =
      this.viewMode === "weights"
        ? "WEIGHT MATRIX (W)"
        : this.viewMode === "deltas"
        ? "LIVE WEIGHT UPDATES (|ΔW|)"
        : "BIASES (b)";
    ctx.fillText(
      `Layer ${this.selectedLayer + 1}: ${modeLabel} [${rows}×${cols}]`,
      12,
      20
    );

    // Compute stats
    let sumW = 0;
    let sumAbsDelta = 0;
    let maxAbsDelta = 0;
    let maxW = -Infinity;
    let minW = Infinity;
    const totalWeights = rows * cols;

    for (let i = 0; i < rows; i++) {
      const wRow = layer.weights[i];
      const dRow = layer.weightDeltas[i];
      for (let j = 0; j < cols; j++) {
        const val = wRow[j];
        const delta = Math.abs(dRow[j]);
        sumW += val;
        sumAbsDelta += delta;
        if (delta > maxAbsDelta) maxAbsDelta = delta;
        if (val > maxW) maxW = val;
        if (val < minW) minW = val;
      }
    }

    const meanW = sumW / totalWeights;
    const meanAbsDelta = sumAbsDelta / totalWeights;

    // Subtitle stats in upper right
    ctx.textAlign = "right";
    ctx.fillStyle = "#00f3ff";
    ctx.fillText(`Mean |ΔW|: ${meanAbsDelta.toExponential(2)}`, w - 12, 20);

    // Grid rendering bounds
    const topMargin = 32;
    const leftMargin = 38;
    const rightMargin = 20;
    const bottomMargin = 24;

    const availableW = w - leftMargin - rightMargin;
    const availableH = h - topMargin - bottomMargin;

    const cellW = Math.min(28, availableW / cols);
    const cellH = Math.min(18, availableH / rows);

    const startX = leftMargin + (availableW - cellW * cols) / 2;
    const startY = topMargin + (availableH - cellH * rows) / 2;

    this.gridConfig = { startX, startY, cellW, cellH, rows, cols };

    // Axis labels: Input indices (cols)
    ctx.fillStyle = "#5c6b84";
    ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    for (let j = 0; j < cols; j++) {
      if (cols > 15 && j % 2 !== 0) continue; // Skip every other label if dense
      ctx.fillText(`j${j}`, startX + (j + 0.5) * cellW, startY - 5);
    }

    // Axis labels: Output indices (rows)
    ctx.textAlign = "right";
    for (let i = 0; i < rows; i++) {
      if (rows > 14 && i % 2 !== 0) continue;
      ctx.fillText(`i${i}`, startX - 6, startY + (i + 0.65) * cellH);
    }

    // Draw Heatmap Cells
    for (let i = 0; i < rows; i++) {
      const wRow = layer.weights[i];
      const dRow = layer.weightDeltas[i];

      for (let j = 0; j < cols; j++) {
        const x = startX + j * cellW;
        const y = startY + i * cellH;

        let color = "#121522";

        if (this.viewMode === "weights") {
          // Divergent map: Positive (Cyan), Zero (Dark), Negative (Magenta)
          const val = wRow[j];
          const clamped = Math.max(-2, Math.min(2, val)) / 2; // -1 to 1
          if (clamped >= 0) {
            const alpha = Math.min(1, Math.max(0.08, clamped));
            color = `rgba(0, 243, 255, ${alpha.toFixed(2)})`;
          } else {
            const alpha = Math.min(1, Math.max(0.08, Math.abs(clamped)));
            color = `rgba(255, 0, 127, ${alpha.toFixed(2)})`;
          }
        } else if (this.viewMode === "deltas") {
          // Heatmap of gradient magnitude: Black -> Purple -> Hot Yellow
          const delta = Math.abs(dRow[j]);
          const normDelta = maxAbsDelta > 0 ? Math.min(1, delta / (maxAbsDelta * 0.8)) : 0;
          if (normDelta > 0.6) {
            color = `rgba(255, 230, 0, ${normDelta.toFixed(2)})`;
          } else if (normDelta > 0.15) {
            color = `rgba(0, 255, 136, ${(normDelta * 1.4).toFixed(2)})`;
          } else {
            color = `rgba(168, 85, 247, ${Math.max(0.1, normDelta * 2).toFixed(2)})`;
          }
        } else {
          // Biases
          const bVal = layer.biases[i];
          const clamped = Math.max(-2, Math.min(2, bVal)) / 2;
          color =
            clamped >= 0
              ? `rgba(0, 243, 255, ${Math.max(0.1, clamped).toFixed(2)})`
              : `rgba(255, 0, 127, ${Math.max(0.1, Math.abs(clamped)).toFixed(2)})`;
        }

        ctx.fillStyle = color;
        ctx.fillRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);

        // Grid border
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.strokeRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
      }
    }

    // Hover Highlight & Tooltip
    if (this.hoverCell) {
      const { row, col, mouseX, mouseY } = this.hoverCell;
      const x = startX + col * cellW;
      const y = startY + row * cellH;

      ctx.save();
      ctx.strokeStyle = "#ffe600";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, cellW, cellH);

      // Tooltip Card
      const weightVal = layer.weights[row][col];
      const deltaVal = layer.weightDeltas[row][col];
      const biasVal = layer.biases[row];

      const tooltipText = [
        `Weight [${row}, ${col}]: ${weightVal.toFixed(5)}`,
        `Last ΔW: ${deltaVal.toExponential(3)}`,
        `Bias [${row}]: ${biasVal.toFixed(5)}`
      ];

      ctx.font = "10px 'JetBrains Mono', monospace";
      let tw = 0;
      for (const t of tooltipText) {
        tw = Math.max(tw, ctx.measureText(t).width);
      }

      const tx = Math.min(w - tw - 24, Math.max(10, mouseX + 12));
      const ty = Math.min(h - 60, Math.max(20, mouseY + 12));

      ctx.fillStyle = "rgba(12, 14, 23, 0.96)";
      ctx.strokeStyle = "#ffe600";
      ctx.lineWidth = 1;
      ctx.fillRect(tx, ty, tw + 16, 56);
      ctx.strokeRect(tx, ty, tw + 16, 56);

      ctx.fillStyle = "#ffe600";
      ctx.textAlign = "left";
      ctx.fillText(tooltipText[0], tx + 8, ty + 16);
      ctx.fillStyle = "#00ff88";
      ctx.fillText(tooltipText[1], tx + 8, ty + 32);
      ctx.fillStyle = "#8a99ad";
      ctx.fillText(tooltipText[2], tx + 8, ty + 48);

      ctx.restore();
    }
  }
}
