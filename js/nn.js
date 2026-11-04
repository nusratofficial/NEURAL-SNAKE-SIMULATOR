/**
 * Deep Neural Network Engine with Adam Optimizer
 * Supports Multi-Layer Perceptrons, LeakyReLU, Linear, Backpropagation,
 * and tracks layer-by-layer weight updates (deltas) for live matrix inspection.
 */

// He initialization for LeakyReLU
function randHe(fanIn) {
  const std = Math.sqrt(2.0 / fanIn);
  // Box-Muller transform for normal distribution
  const u1 = Math.random() || 1e-7;
  const u2 = Math.random() || 1e-7;
  const normal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return normal * std;
}

export class DenseLayer {
  constructor(inputSize, outputSize, activation = "leaky_relu") {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = activation;

    // Weights matrix: shape [outputSize][inputSize]
    this.weights = [];
    // Biases vector: shape [outputSize]
    this.biases = [];

    // Tracks change from last update for matrix inspector
    this.weightDeltas = [];
    this.biasDeltas = [];

    // Adam optimizer moments
    this.m_w = [];
    this.v_w = [];
    this.m_b = [];
    this.v_b = [];

    this.initWeights();
  }

  initWeights() {
    this.weights = [];
    this.biases = [];
    this.weightDeltas = [];
    this.biasDeltas = [];
    this.m_w = [];
    this.v_w = [];
    this.m_b = [];
    this.v_b = [];

    for (let i = 0; i < this.outputSize; i++) {
      this.weights[i] = new Float32Array(this.inputSize);
      this.weightDeltas[i] = new Float32Array(this.inputSize);
      this.m_w[i] = new Float32Array(this.inputSize);
      this.v_w[i] = new Float32Array(this.inputSize);

      for (let j = 0; j < this.inputSize; j++) {
        this.weights[i][j] = randHe(this.inputSize);
      }
      this.biases[i] = 0.01;
      this.biasDeltas[i] = 0.0;
      this.m_b[i] = 0.0;
      this.v_b[i] = 0.0;
    }
  }

  forward(inputs) {
    // inputs: Array or Float32Array of length inputSize
    const z = new Float32Array(this.outputSize);
    const a = new Float32Array(this.outputSize);

    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.biases[i];
      const row = this.weights[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += row[j] * inputs[j];
      }
      z[i] = sum;

      if (this.activation === "leaky_relu") {
        a[i] = sum > 0 ? sum : sum * 0.01;
      } else if (this.activation === "relu") {
        a[i] = sum > 0 ? sum : 0;
      } else if (this.activation === "tanh") {
        a[i] = Math.tanh(sum);
      } else {
        // linear
        a[i] = sum;
      }
    }

    return { z, a };
  }

  activationDerivative(z, a) {
    const da = new Float32Array(this.outputSize);
    for (let i = 0; i < this.outputSize; i++) {
      if (this.activation === "leaky_relu") {
        da[i] = z[i] > 0 ? 1.0 : 0.01;
      } else if (this.activation === "relu") {
        da[i] = z[i] > 0 ? 1.0 : 0.0;
      } else if (this.activation === "tanh") {
        da[i] = 1.0 - a[i] * a[i];
      } else {
        da[i] = 1.0;
      }
    }
    return da;
  }
}

export class NeuralNetwork {
  constructor(layerSizes = [12, 18, 12, 3]) {
    this.layerSizes = layerSizes;
    this.layers = [];
    this.t = 0; // Adam time step counter

    // Construct layers
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const isLast = i === layerSizes.length - 2;
      const activation = isLast ? "linear" : "leaky_relu";
      this.layers.push(new DenseLayer(layerSizes[i], layerSizes[i + 1], activation));
    }

    // Cache latest activations for visualization
    this.lastActivations = [];
    this.lastInputs = null;
  }

  forward(inputs) {
    this.lastInputs = inputs;
    this.lastActivations = [inputs];

    let current = inputs;
    const forwardCache = []; // { inputs, z, a } for backpropagation

    for (const layer of this.layers) {
      const { z, a } = layer.forward(current);
      forwardCache.push({ inputs: current, z, a });
      this.lastActivations.push(a);
      current = a;
    }

    return { output: current, cache: forwardCache };
  }

  predict(inputs) {
    return this.forward(inputs).output;
  }

  /**
   * Train network on a batch of experience samples using Adam optimizer
   * @param {Array} samples: array of { state, action, targetQ }
   * @param {number} learningRate: Adam alpha
   * @param {number} beta1: 0.9
   * @param {number} beta2: 0.999
   * @param {number} eps: 1e-8
   */
  trainBatch(samples, learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    if (!samples || samples.length === 0) return 0;
    this.t++;
    const batchSize = samples.length;

    // Accumulators for gradients
    const gradW = this.layers.map(l =>
      Array.from({ length: l.outputSize }, () => new Float32Array(l.inputSize))
    );
    const gradB = this.layers.map(l => new Float32Array(l.outputSize));

    let totalLoss = 0;

    for (const sample of samples) {
      const { output, cache } = this.forward(sample.state);

      // Compute loss and dZ for output layer
      // Smooth L1 (Huber) or MSE
      const dZ_out = new Float32Array(output.length);
      const action = sample.action;
      const error = output[action] - sample.targetQ;

      // Huber loss gradient clipping for RL stability
      let clippedError = error;
      if (Math.abs(error) > 1.0) {
        clippedError = Math.sign(error);
        totalLoss += Math.abs(error) - 0.5;
      } else {
        totalLoss += 0.5 * error * error;
      }
      dZ_out[action] = clippedError;

      // Backpropagation through layers
      let current_dZ = dZ_out;

      for (let l = this.layers.length - 1; l >= 0; l--) {
        const layer = this.layers[l];
        const layerCache = cache[l];
        const layerIn = layerCache.inputs;

        // Gradient w.r.t weights and biases for layer l
        for (let i = 0; i < layer.outputSize; i++) {
          const dzi = current_dZ[i];
          gradB[l][i] += dzi;
          const gwRow = gradW[l][i];
          for (let j = 0; j < layer.inputSize; j++) {
            gwRow[j] += dzi * layerIn[j];
          }
        }

        // Propagate gradient to preceding layer (if not input layer)
        if (l > 0) {
          const prevLayer = this.layers[l - 1];
          const prevCache = cache[l - 1];
          const prevDeriv = prevLayer.activationDerivative(prevCache.z, prevCache.a);
          const next_dZ = new Float32Array(layer.inputSize);

          for (let j = 0; j < layer.inputSize; j++) {
            let sum = 0;
            for (let i = 0; i < layer.outputSize; i++) {
              sum += layer.weights[i][j] * current_dZ[i];
            }
            next_dZ[j] = sum * prevDeriv[j];
          }
          current_dZ = next_dZ;
        }
      }
    }

    // Adam Parameter Updates
    const beta1_t = Math.pow(beta1, this.t);
    const beta2_t = Math.pow(beta2, this.t);
    const alpha_t = learningRate * Math.sqrt(1.0 - beta2_t) / (1.0 - beta1_t);

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const gw = gradW[l];
      const gb = gradB[l];

      for (let i = 0; i < layer.outputSize; i++) {
        // Bias update
        const g_b = gb[i] / batchSize;
        layer.m_b[i] = beta1 * layer.m_b[i] + (1.0 - beta1) * g_b;
        layer.v_b[i] = beta2 * layer.v_b[i] + (1.0 - beta2) * (g_b * g_b);
        const m_b_hat = layer.m_b[i] / (1.0 - beta1_t);
        const v_b_hat = layer.v_b[i] / (1.0 - beta2_t);
        const delta_b = -learningRate * m_b_hat / (Math.sqrt(v_b_hat) + eps);
        layer.biases[i] += delta_b;
        layer.biasDeltas[i] = delta_b;

        // Weights update
        const wRow = layer.weights[i];
        const deltaRow = layer.weightDeltas[i];
        const mwRow = layer.m_w[i];
        const vwRow = layer.v_w[i];
        const gwRow = gw[i];

        for (let j = 0; j < layer.inputSize; j++) {
          const g_w = gwRow[j] / batchSize;
          mwRow[j] = beta1 * mwRow[j] + (1.0 - beta1) * g_w;
          vwRow[j] = beta2 * vwRow[j] + (1.0 - beta2) * (g_w * g_w);
          const m_w_hat = mwRow[j] / (1.0 - beta1_t);
          const v_w_hat = vwRow[j] / (1.0 - beta2_t);
          const delta_w = -learningRate * m_w_hat / (Math.sqrt(v_w_hat) + eps);
          wRow[j] += delta_w;
          deltaRow[j] = delta_w;
        }
      }
    }

    return totalLoss / batchSize;
  }

  /**
   * Clone network architecture and weights
   */
  clone() {
    const copy = new NeuralNetwork(this.layerSizes);
    copy.copyFrom(this);
    return copy;
  }

  /**
   * Copy weights from another network (e.g. Target Network sync)
   */
  copyFrom(source, tau = 1.0) {
    for (let l = 0; l < this.layers.length; l++) {
      const targetLayer = this.layers[l];
      const sourceLayer = source.layers[l];

      for (let i = 0; i < targetLayer.outputSize; i++) {
        if (tau === 1.0) {
          targetLayer.biases[i] = sourceLayer.biases[i];
        } else {
          targetLayer.biases[i] = tau * sourceLayer.biases[i] + (1.0 - tau) * targetLayer.biases[i];
        }

        const tRow = targetLayer.weights[i];
        const sRow = sourceLayer.weights[i];
        for (let j = 0; j < targetLayer.inputSize; j++) {
          if (tau === 1.0) {
            tRow[j] = sRow[j];
          } else {
            tRow[j] = tau * sRow[j] + (1.0 - tau) * tRow[j];
          }
        }
      }
    }
  }

  /**
   * Export network weights to JSON
   */
  toJSON() {
    return {
      layerSizes: this.layerSizes,
      t: this.t,
      layers: this.layers.map(l => ({
        inputSize: l.inputSize,
        outputSize: l.outputSize,
        activation: l.activation,
        weights: l.weights.map(row => Array.from(row)),
        biases: Array.from(l.biases)
      }))
    };
  }

  /**
   * Load network weights from JSON
   */
  fromJSON(data) {
    this.layerSizes = data.layerSizes;
    this.t = data.t || 100;
    this.layers = [];

    for (const lData of data.layers) {
      const layer = new DenseLayer(lData.inputSize, lData.outputSize, lData.activation);
      for (let i = 0; i < lData.outputSize; i++) {
        layer.biases[i] = lData.biases[i];
        for (let j = 0; j < lData.inputSize; j++) {
          layer.weights[i][j] = lData.weights[i][j];
        }
      }
      this.layers.push(layer);
    }
  }
}

