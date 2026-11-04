/**
 * Snake Game Environment for Reinforcement Learning
 * Grid-based world with relative sensory perception and reward shaping.
 */
export class SnakeGame {
  constructor(gridWidth = 20, gridHeight = 20) {
    this.width = gridWidth;
    this.height = gridHeight;
    this.reset();
  }

  reset() {
    // Start snake in the center
    const startX = Math.floor(this.width / 2);
    const startY = Math.floor(this.height / 2);

    this.head = { x: startX, y: startY };
    this.body = [
      { x: startX, y: startY },
      { x: startX - 1, y: startY },
      { x: startX - 2, y: startY }
    ];

    // Directions: 0: Up (0, -1), 1: Right (1, 0), 2: Down (0, 1), 3: Left (-1, 0)
    this.direction = { x: 1, y: 0 }; // Moving right initially
    this.score = 0;
    this.steps = 0;
    this.stepsWithoutFood = 0;
    this.maxStepsWithoutFood = 120; // Will scale with snake length
    this.isGameOver = false;

    this.placeFood();
    return this.getState();
  }

  placeFood() {
    const occupied = new Set();
    for (const seg of this.body) {
      occupied.add(`${seg.x},${seg.y}`);
    }

    const freeSpots = [];
    for (let x = 0; x < this.width; x++) {
      for (let y = 0; y < this.height; y++) {
        if (!occupied.has(`${x},${y}`)) {
          freeSpots.push({ x, y });
        }
      }
    }

    if (freeSpots.length === 0) {
      // Game won (filled entire grid!)
      this.isGameOver = true;
      return;
    }

    const spot = freeSpots[Math.floor(Math.random() * freeSpots.length)];
    this.food = spot;
  }

  isCollision(pt) {
    // Wall collision
    if (pt.x < 0 || pt.x >= this.width || pt.y < 0 || pt.y >= this.height) {
      return true;
    }
    // Body collision (ignore tail tip if it will move, but check body)
    for (let i = 0; i < this.body.length - 1; i++) {
      if (this.body[i].x === pt.x && this.body[i].y === pt.y) {
        return true;
      }
    }
    return false;
  }

  getHeadingVectors() {
    const dir = this.direction;
    // Straight: [dir.x, dir.y]
    // Relative Right (clockwise in screen coords where Y is down): [-dir.y, dir.x]
    // Relative Left (counter-clockwise): [dir.y, -dir.x]
    return {
      straight: { x: dir.x, y: dir.y },
      right: { x: -dir.y, y: dir.x },
      left: { x: dir.y, y: -dir.x }
    };
  }

  /**
   * 12-dimensional continuous/binary state representation
   */
  getState() {
    const { straight, right, left } = this.getHeadingVectors();
    const head = this.head;
    const food = this.food || { x: 0, y: 0 };

    const ptStraight = { x: head.x + straight.x, y: head.y + straight.y };
    const ptRight = { x: head.x + right.x, y: head.y + right.y };
    const ptLeft = { x: head.x + left.x, y: head.y + left.y };

    const dangerStraight = this.isCollision(ptStraight) ? 1.0 : 0.0;
    const dangerRight = this.isCollision(ptRight) ? 1.0 : 0.0;
    const dangerLeft = this.isCollision(ptLeft) ? 1.0 : 0.0;

    // Current moving direction
    const dirUp = this.direction.y === -1 ? 1.0 : 0.0;
    const dirRight = this.direction.x === 1 ? 1.0 : 0.0;
    const dirDown = this.direction.y === 1 ? 1.0 : 0.0;
    const dirLeft = this.direction.x === -1 ? 1.0 : 0.0;

    // Relative food position
    const foodUp = food.y < head.y ? 1.0 : 0.0;
    const foodRight = food.x > head.x ? 1.0 : 0.0;
    const foodDown = food.y > head.y ? 1.0 : 0.0;
    const foodLeft = food.x < head.x ? 1.0 : 0.0;

    // Normalized Euclidean distance to food
    const maxDist = Math.hypot(this.width, this.height);
    const distToFood = Math.hypot(food.x - head.x, food.y - head.y) / maxDist;

    return [
      dangerStraight,
      dangerRight,
      dangerLeft,
      dirUp,
      dirRight,
      dirDown,
      dirLeft,
      foodUp,
      foodRight,
      foodDown,
      foodLeft,
      distToFood
    ];
  }

  /**
   * Feature labels for neural network input visualizer
   */
  static get featureNames() {
    return [
      "Danger Straight",
      "Danger Right",
      "Danger Left",
      "Dir Up",
      "Dir Right",
      "Dir Down",
      "Dir Left",
      "Food Up",
      "Food Right",
      "Food Down",
      "Food Left",
      "Distance to Food"
    ];
  }

  static get actionNames() {
    return ["Go Straight", "Turn Right", "Turn Left"];
  }

  /**
   * Perform one step in the environment
   * @param {number} action: 0 = straight, 1 = turn right, 2 = turn left
   * @param {Object} rewardWeights: configurable reward weights
   */
  step(action = 0, rewardWeights = { food: 10, death: -10, closer: 0.1, farther: -0.15, step: -0.01 }) {
    if (this.isGameOver) {
      return { state: this.getState(), reward: 0, done: true, score: this.score };
    }

    this.steps++;
    this.stepsWithoutFood++;

    const { straight, right, left } = this.getHeadingVectors();
    if (action === 0) {
      this.direction = straight;
    } else if (action === 1) {
      this.direction = right;
    } else if (action === 2) {
      this.direction = left;
    }

    const prevDist = Math.hypot(this.food.x - this.head.x, this.food.y - this.head.y);

    // Compute new head location
    const newHead = {
      x: this.head.x + this.direction.x,
      y: this.head.y + this.direction.y
    };

    let reward = rewardWeights.step || -0.01;
    let done = false;

    // Check collision with walls or body
    if (this.isCollision(newHead)) {
      this.isGameOver = true;
      done = true;
      reward = rewardWeights.death || -10;
      return { state: this.getState(), reward, done, score: this.score, reason: "collision" };
    }

    // Check hunger timeout (prevent infinite loops)
    const hungerLimit = this.maxStepsWithoutFood + this.body.length * 4;
    if (this.stepsWithoutFood > hungerLimit) {
      this.isGameOver = true;
      done = true;
      reward = rewardWeights.death || -10;
      return { state: this.getState(), reward, done, score: this.score, reason: "starvation" };
    }

    // Move snake
    this.head = newHead;
    this.body.unshift({ x: newHead.x, y: newHead.y });

    // Check food eaten
    if (newHead.x === this.food.x && newHead.y === this.food.y) {
      this.score++;
      this.stepsWithoutFood = 0;
      reward = rewardWeights.food || 10;
      this.placeFood();
    } else {
      // Remove tail segment if food not eaten
      this.body.pop();

      // Distance reward shaping
      const newDist = Math.hypot(this.food.x - this.head.x, this.food.y - this.head.y);
      if (newDist < prevDist) {
        reward += rewardWeights.closer || 0.1;
      } else {
        reward += rewardWeights.farther || -0.15;
      }
    }

    return {
      state: this.getState(),
      reward,
      done,
      score: this.score
    };
  }
}

