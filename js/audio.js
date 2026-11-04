/**
 * Futuristic Web Audio Synthesizer
 * Generates cyber audio effects using native Web Audio API oscillators and gain envelopes.
 */

export class CyberAudio {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.hasInteracted = false;

    // Check localStorage preference
    const saved = localStorage.getItem("snake_ai_audio_enabled");
    if (saved !== null) {
      this.enabled = saved === "true";
    }
  }

  init() {
    if (!this.ctx && typeof window !== "undefined") {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
        this.hasInteracted = true;
      }
    }
  }

  toggle() {
    this.enabled = !this.enabled;
    localStorage.setItem("snake_ai_audio_enabled", this.enabled.toString());
    if (this.enabled) {
      this.init();
      this.playTone(440, "sine", 0.1, 0.1);
    }
    return this.enabled;
  }

  playTone(freq, type = "sine", duration = 0.15, volume = 0.15) {
    if (!this.enabled) return;
    if (!this.ctx) this.init();
    if (!this.ctx) return;

    if (this.ctx.state === "suspended") {
      this.ctx.resume();
    }

    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);

      gain.gain.setValueAtTime(volume, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {
      // Audio autoplay policy catch
    }
  }

  playFood() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    const notes = [523.25, 659.25, 783.99, 1046.5]; // C5, E5, G5, C6
    notes.forEach((freq, idx) => {
      setTimeout(() => {
        this.playTone(freq, "triangle", 0.12, 0.1);
      }, idx * 40);
    });
  }

  playTurn() {
    if (!this.enabled) return;
    this.playTone(280, "sine", 0.04, 0.03);
  }

  playCrash() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    try {
      const now = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(120, now);
      osc.frequency.exponentialRampToValueAtTime(30, now + 0.35);

      gain.gain.setValueAtTime(0.25, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(now + 0.35);
    } catch (e) {}
  }

  playHighScore() {
    if (!this.enabled) return;
    const chord = [523.25, 659.25, 783.99, 987.77, 1046.5];
    chord.forEach((f, i) => {
      setTimeout(() => {
        this.playTone(f, "sine", 0.25, 0.12);
      }, i * 70);
    });
  }
}

