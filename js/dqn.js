/**
 * Deep Q-Learning (DQN) Agent with Double DQN and Experience Replay
 */
import { NeuralNetwork } from "./nn.js";

export class ReplayBuffer {
  constructor(capacity = 25000) {
    this.capacity = capacity;
    this.buffer = [];
    this.pointer = 0;
  }

  push(state, action, reward, nextState, done) {
    const item = { state, action, reward, nextState, done };
    if (this.buffer.length < this.capacity) {
      this.buffer.push(item);
    } else {
      this.buffer[this.pointer] = item;
    }
    this.pointer = (this.pointer + 1) % this.capacity;
  }

  sample(batchSize) {
    const samples = [];
    const len = this.buffer.length;
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * len);
      samples.push(this.buffer[idx]);
    }
    return samples;
  }

  size() {
    return this.buffer.length;
  }

  clear() {
    this.buffer = [];
    this.pointer = 0;
  }
}

export class DQNAgent {
  constructor(options = {}) {
    this.layerSizes = options.layerSizes || [12, 18, 12, 3];
    this.policyNet = new NeuralNetwork(this.layerSizes);
    this.targetNet = this.policyNet.clone();

    // Replay buffer
    this.bufferSize = options.bufferSize || 30000;
    this.memory = new ReplayBuffer(this.bufferSize);

    // Hyperparameters
    this.gamma = options.gamma ?? 0.95;
    this.epsilon = options.epsilon ?? 1.0;
    this.minEpsilon = options.minEpsilon ?? 0.01;
    this.epsilonDecay = options.epsilonDecay ?? 0.995;
    this.learningRate = options.learningRate ?? 0.001;
    this.batchSize = options.batchSize ?? 64;
    this.tau = options.tau ?? 0.01; // Soft update parameter

    this.steps = 0;
    this.episodes = 0;
    this.latestLoss = 0;
    this.latestQValues = [0, 0, 0];
    this.isTraining = true;
  }

  /**
   * Select an action using epsilon-greedy policy
   */
  act(state, forceExploit = false) {
    const qValues = this.policyNet.predict(state);
    this.latestQValues = Array.from(qValues);

    if (!forceExploit && this.isTraining && Math.random() < this.epsilon) {
      // Explore: random action
      return {
        action: Math.floor(Math.random() * this.layerSizes[this.layerSizes.length - 1]),
        isExploration: true,
        qValues: this.latestQValues
      };
    }

    // Exploit: argmax Q
    let bestAction = 0;
    let maxQ = -Infinity;
    for (let a = 0; a < qValues.length; a++) {
      if (qValues[a] > maxQ) {
        maxQ = qValues[a];
        bestAction = a;
      }
    }

    return {
      action: bestAction,
      isExploration: false,
      qValues: this.latestQValues
    };
  }

  /**
   * Store transition in experience replay
   */
  remember(state, action, reward, nextState, done) {
    this.memory.push(state, action, reward, nextState, done);
    this.steps++;
  }

  /**
   * Train step on mini-batch from replay buffer (Double DQN)
   */
  train() {
    if (this.memory.size() < this.batchSize) {
      return 0;
    }

    const batch = this.memory.sample(this.batchSize);
    const trainingSamples = [];

    for (const exp of batch) {
      let targetQ = exp.reward;

      if (!exp.done) {
        // Double DQN target calculation:
        // Action selected by policy net
        const nextPolicyQ = this.policyNet.predict(exp.nextState);
        let bestNextAction = 0;
        let maxNextQ = -Infinity;
        for (let a = 0; a < nextPolicyQ.length; a++) {
          if (nextPolicyQ[a] > maxNextQ) {
            maxNextQ = nextPolicyQ[a];
            bestNextAction = a;
          }
        }

        // Q-value evaluated by target net for chosen action
        const nextTargetQ = this.targetNet.predict(exp.nextState);
        targetQ = exp.reward + this.gamma * nextTargetQ[bestNextAction];
      }

      trainingSamples.push({
        state: exp.state,
        action: exp.action,
        targetQ
      });
    }

    // Update policy network
    const loss = this.policyNet.trainBatch(trainingSamples, this.learningRate);
    this.latestLoss = loss;

    // Soft update target network: theta_target = tau * theta + (1 - tau) * theta_target
    this.targetNet.copyFrom(this.policyNet, this.tau);

    return loss;
  }

  /**
   * Decay epsilon exploration rate after an episode
   */
  decayEpsilon() {
    this.episodes++;
    if (this.epsilon > this.minEpsilon) {
      this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
    }
  }

  /**
   * Reset brain to fresh random weights
   */
  reset() {
    this.policyNet = new NeuralNetwork(this.layerSizes);
    this.targetNet = this.policyNet.clone();
    this.memory.clear();
    this.epsilon = 1.0;
    this.steps = 0;
    this.episodes = 0;
    this.latestLoss = 0;
  }
}

