/**
 * Offline Training Script for Snake DQN Agent
 * Trains neural network, tracks score progression, and outputs pretrained weights.
 */
import fs from "fs";
import { SnakeGame } from "../js/snake.js";
import { DQNAgent } from "../js/dqn.js";

// Helper: Simple safe greedy heuristic for imitation pre-warming
function getHeuristicAction(game) {
  const { straight, right, left } = game.getHeadingVectors();
  const head = game.head;
  const food = game.food;

  const ptStraight = { x: head.x + straight.x, y: head.y + straight.y };
  const ptRight = { x: head.x + right.x, y: head.y + right.y };
  const ptLeft = { x: head.x + left.x, y: head.y + left.y };

  const safeStraight = !game.isCollision(ptStraight);
  const safeRight = !game.isCollision(ptRight);
  const safeLeft = !game.isCollision(ptLeft);

  const candidates = [];
  if (safeStraight) {
    const dist = Math.hypot(food.x - ptStraight.x, food.y - ptStraight.y);
    candidates.push({ action: 0, dist });
  }
  if (safeRight) {
    const dist = Math.hypot(food.x - ptRight.x, food.y - ptRight.y);
    candidates.push({ action: 1, dist });
  }
  if (safeLeft) {
    const dist = Math.hypot(food.x - ptLeft.x, food.y - ptLeft.y);
    candidates.push({ action: 2, dist });
  }

  if (candidates.length === 0) return 0; // trapped
  candidates.sort((a, b) => a.dist - b.dist);
  return candidates[0].action;
}

console.log("Starting Snake DQN Training...");
const game = new SnakeGame(20, 20);
const agent = new DQNAgent({
  layerSizes: [12, 18, 12, 3],
  learningRate: 0.001,
  gamma: 0.95,
  batchSize: 64,
  epsilon: 1.0,
  epsilonDecay: 0.99,
  minEpsilon: 0.02,
  bufferSize: 30000
});

// Phase 1: Pre-populate replay memory with 3000 heuristic steps
console.log("Pre-populating replay memory with guided experiences...");
let warmupSteps = 0;
while (warmupSteps < 3000) {
  game.reset();
  let state = game.getState();
  let done = false;
  while (!done && warmupSteps < 3000) {
    const action = getHeuristicAction(game);
    const { state: nextState, reward, done: isDone } = game.step(action);
    agent.remember(state, action, reward, nextState, isDone);
    state = nextState;
    done = isDone;
    warmupSteps++;
  }
}
console.log(`Replay memory initialized with ${agent.memory.size()} transitions.`);

// Pre-train policy network on warmed memory
console.log("Pre-training policy network on memory buffer...");
for (let i = 0; i < 200; i++) {
  agent.train();
}

// Phase 2: Active RL Training Episodes
const totalEpisodes = 300;
let highScore = 0;
let recentScores = [];

for (let ep = 1; ep <= totalEpisodes; ep++) {
  game.reset();
  let state = game.getState();
  let done = false;
  let totalReward = 0;

  while (!done) {
    // Epsilon-greedy action
    const { action } = agent.act(state);
    const { state: nextState, reward, done: isDone, score } = game.step(action);

    agent.remember(state, action, reward, nextState, isDone);
    agent.train();

    state = nextState;
    done = isDone;
    totalReward += reward;
  }

  agent.decayEpsilon();
  const score = game.score;
  if (score > highScore) highScore = score;
  recentScores.push(score);
  if (recentScores.length > 50) recentScores.shift();

  const avgScore = (recentScores.reduce((a, b) => a + b, 0) / recentScores.length).toFixed(2);

  if (ep % 25 === 0 || ep === totalEpisodes) {
    console.log(
      `Episode ${ep}/${totalEpisodes} | Score: ${score} | High: ${highScore} | Avg(50): ${avgScore} | Epsilon: ${agent.epsilon.toFixed(3)} | Loss: ${agent.latestLoss.toFixed(4)}`
    );
  }
}

// Export trained weights to js/pretrained.js
const weightsJSON = agent.policyNet.toJSON();
const fileContent = `/**
 * Pre-trained Deep Q-Network Weights for Snake AI
 * Generated via Double DQN training with experience replay.
 */
export const PRETRAINED_WEIGHTS = ${JSON.stringify(weightsJSON, null, 2)};
`;

fs.writeFileSync("./js/pretrained.js", fileContent, "utf8");
console.log("Pretrained weights successfully saved to js/pretrained.js!");

