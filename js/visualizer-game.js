/**
 * Cyberpunk Game Visualizer for Snake AI
 * Features neon glow, particle bursts, sensory ray visualization, and smooth rendering.
 */

export class GameVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.particles = [];
    this.showSensors = true;

    // Handle high DPI displays
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;

    this.canvas.width = (rect.width || 440) * dpr;
    this.canvas.height = (rect.height || 440) * dpr;
    this.ctx.scale(dpr, dpr);
    this.displayW = rect.width || 440;
    this.displayH = rect.height || 440;
  }

  spawnFoodParticles(x, y, color = "#00f3ff") {
    for (let i = 0; i < 18; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 3.5 + 1.2;
      this.particles.push({
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1.0,
        decay: Math.random() * 0.03 + 0.02,
        size: Math.random() * 3 + 2,
        color
      });
    }
  }

  spawnDeathParticles(x, y) {
    for (let i = 0; i < 28; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 4.5 + 1.0;
      this.particles.push({
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1.0,
        decay: Math.random() * 0.025 + 0.015,
        size: Math.random() * 4 + 2,
        color: Math.random() > 0.5 ? "#ff007f" : "#ff3344"
      });
    }
  }

  updateParticles() {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.96;
      p.vy *= 0.96;
      p.life -= p.decay;
      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }
  }

  render(game, time = 0) {
    const ctx = this.ctx;
    const w = this.displayW;
    const h = this.displayH;

    // Clear background
    ctx.fillStyle = "#0c0e17";
    ctx.fillRect(0, 0, w, h);

    const cellW = w / game.width;
    const cellH = h / game.height;

    // Draw grid lines
    ctx.strokeStyle = "rgba(0, 243, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= game.width; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cellW, 0);
      ctx.lineTo(x * cellW, h);
      ctx.stroke();
    }
    for (let y = 0; y <= game.height; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * cellH);
      ctx.lineTo(w, y * cellH);
      ctx.stroke();
    }

    // Sensory rays from snake head (visualizing what the AI sees)
    if (this.showSensors && game.head && !game.isGameOver) {
      this.drawSensoryRays(game, cellW, cellH);
    }

    // Draw Food
    if (game.food) {
      const fx = (game.food.x + 0.5) * cellW;
      const fy = (game.food.y + 0.5) * cellH;
      const pulse = Math.sin(time * 0.006) * 2;
      const foodRadius = Math.max(2, cellW * 0.38 + pulse);

      ctx.save();
      ctx.shadowColor = "#ff007f";
      ctx.shadowBlur = 15;

      // Outer glow ring
      ctx.strokeStyle = "rgba(255, 0, 127, 0.6)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(fx, fy, foodRadius + 3, 0, Math.PI * 2);
      ctx.stroke();

      // Food core
      const foodGrad = ctx.createRadialGradient(fx - 2, fy - 2, 1, fx, fy, foodRadius);
      foodGrad.addColorStop(0, "#ffffff");
      foodGrad.addColorStop(0.4, "#ff4099");
      foodGrad.addColorStop(1, "#ff007f");
      ctx.fillStyle = foodGrad;
      ctx.beginPath();
      ctx.arc(fx, fy, foodRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Draw Snake Body
    if (game.body && game.body.length > 0) {
      const len = game.body.length;

      for (let i = len - 1; i >= 0; i--) {
        const seg = game.body[i];
        const segX = seg.x * cellW;
        const segY = seg.y * cellH;
        const padding = 1.5;

        ctx.save();
        if (i === 0) {
          // Snake Head
          ctx.shadowColor = "#00f3ff";
          ctx.shadowBlur = 14;

          const headGrad = ctx.createLinearGradient(segX, segY, segX + cellW, segY + cellH);
          headGrad.addColorStop(0, "#e0ffff");
          headGrad.addColorStop(1, "#00f3ff");
          ctx.fillStyle = headGrad;

          this.drawRoundedRect(
            ctx,
            segX + padding,
            segY + padding,
            cellW - padding * 2,
            cellH - padding * 2,
            5
          );
          ctx.fill();

          // Snake Eyes
          ctx.fillStyle = "#0c0e17";
          const eyeSize = Math.max(2, cellW * 0.18);
          const cx = segX + cellW / 2;
          const cy = segY + cellH / 2;
          const dx = game.direction.x;
          const dy = game.direction.y;

          // Eye offset relative to direction
          const orthoX = -dy;
          const orthoY = dx;
          const eyeDist = cellW * 0.22;
          const eyeForward = cellW * 0.2;

          ctx.beginPath();
          ctx.arc(
            cx + dx * eyeForward + orthoX * eyeDist,
            cy + dy * eyeForward + orthoY * eyeDist,
            eyeSize,
            0,
            Math.PI * 2
          );
          ctx.arc(
            cx + dx * eyeForward - orthoX * eyeDist,
            cy + dy * eyeForward - orthoY * eyeDist,
            eyeSize,
            0,
            Math.PI * 2
          );
          ctx.fill();
        } else {
          // Body segments with color gradient from cyan to electric purple
          const t = i / len;
          const r = Math.round(0 * (1 - t) + 168 * t);
          const g = Math.round(243 * (1 - t) + 85 * t);
          const b = Math.round(255 * (1 - t) + 247 * t);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.shadowColor = `rgba(${r}, ${g}, ${b}, 0.4)`;
          ctx.shadowBlur = 6;

          this.drawRoundedRect(
            ctx,
            segX + padding,
            segY + padding,
            cellW - padding * 2,
            cellH - padding * 2,
            3
          );
          ctx.fill();
        }
        ctx.restore();
      }
    }

    // Render Particles
    this.updateParticles();
    for (const p of this.particles) {
      ctx.save();
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.life;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Game Over Overlay
    if (game.isGameOver) {
      ctx.fillStyle = "rgba(10, 12, 20, 0.7)";
      ctx.fillRect(0, 0, w, h);

      ctx.save();
      ctx.shadowColor = "#ff007f";
      ctx.shadowBlur = 16;
      ctx.font = "bold 24px 'JetBrains Mono', 'Segoe UI', monospace";
      ctx.fillStyle = "#ff007f";
      ctx.textAlign = "center";
      ctx.fillText("AGENT TERMINATED", w / 2, h / 2 - 10);

      ctx.font = "14px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#8a99ad";
      ctx.fillText("Reinforcement loop iterating...", w / 2, h / 2 + 18);
      ctx.restore();
    }
  }

  drawSensoryRays(game, cellW, cellH) {
    const ctx = this.ctx;
    const hx = (game.head.x + 0.5) * cellW;
    const hy = (game.head.y + 0.5) * cellH;

    const { straight, right, left } = game.getHeadingVectors();
    const dirs = [
      { vec: straight, label: "S" },
      { vec: right, label: "R" },
      { vec: left, label: "L" }
    ];

    for (const d of dirs) {
      const targetPt = {
        x: game.head.x + d.vec.x,
        y: game.head.y + d.vec.y
      };
      const blocked = game.isCollision(targetPt);
      const targetScreenX = (targetPt.x + 0.5) * cellW;
      const targetScreenY = (targetPt.y + 0.5) * cellH;

      ctx.save();
      ctx.strokeStyle = blocked ? "rgba(255, 0, 127, 0.7)" : "rgba(0, 255, 136, 0.45)";
      ctx.lineWidth = blocked ? 2.2 : 1.2;
      ctx.setLineDash([3, 3]);

      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(targetScreenX, targetScreenY);
      ctx.stroke();

      // Sensory probe head
      ctx.fillStyle = blocked ? "#ff007f" : "#00ff88";
      ctx.beginPath();
      ctx.arc(targetScreenX, targetScreenY, blocked ? 3.5 : 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }
}

