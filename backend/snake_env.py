"""
Snake Game Environment for Reinforcement Learning in Python.
Grid-based world with relative sensory perception, danger detection,
reachable space analysis (BFS), and reward shaping.
"""
import math
import random
from collections import deque
from typing import Dict, List, Tuple, Any

class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> List[float]:
        start_x = self.width // 2
        start_y = self.height // 2

        self.head = (start_x, start_y)
        self.body = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ]
        # Direction: (dx, dy)
        self.direction = (1, 0) # Moving right initially
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.max_steps_without_food = 250
        self.is_game_over = False

        self.place_food()
        return self.get_state()

    def place_food(self):
        occupied = set(self.body)
        free_spots = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]

        if not free_spots:
            # Filled grid
            self.is_game_over = True
            return

        self.food = random.choice(free_spots)

    def is_collision(self, pt: Tuple[int, int]) -> bool:
        x, y = pt
        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # Self collision (exclude tail tip if it will move, but check body)
        if pt in self.body[:-1]:
            return True
        return False

    def get_heading_vectors(self) -> Dict[str, Tuple[int, int]]:
        dx, dy = self.direction
        return {
            "straight": (dx, dy),
            "right": (-dy, dx),
            "left": (dy, -dx)
        }

    def analyze_action_space(self, action: int) -> Dict[str, Any]:
        """
        Fast BFS analysis of space and tail reachability if action is taken.
        """
        headings = self.get_heading_vectors()
        dirs = [headings["straight"], headings["right"], headings["left"]]
        chosen_dir = dirs[action]
        next_head = (self.head[0] + chosen_dir[0], self.head[1] + chosen_dir[1])

        if self.is_collision(next_head):
            return {
                "safe": False,
                "collision": True,
                "space": 0,
                "can_reach_tail": False,
                "is_trap": True
            }

        occupied = set(self.body[:-1])
        occupied.add(next_head)

        queue = deque([next_head])
        visited = {next_head}
        tail = self.body[-1]
        can_reach_tail = False

        while queue:
            curr = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = curr[0] + dx
                ny = curr[1] + dy
                neighbor = (nx, ny)
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if neighbor == tail:
                        can_reach_tail = True
                    if neighbor not in occupied and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        space = len(visited)
        snake_len = len(self.body)
        is_trap = (space < snake_len and not can_reach_tail)

        return {
            "safe": not is_trap,
            "collision": False,
            "space": space,
            "can_reach_tail": can_reach_tail,
            "is_trap": is_trap
        }

    def get_state(self) -> List[float]:
        headings = self.get_heading_vectors()
        straight = headings["straight"]
        right = headings["right"]
        left = headings["left"]

        hx, hy = self.head
        fx, fy = self.food if hasattr(self, "food") and self.food else (0, 0)

        pt_straight = (hx + straight[0], hy + straight[1])
        pt_right = (hx + right[0], hy + right[1])
        pt_left = (hx + left[0], hy + left[1])

        danger_straight = 1.0 if self.is_collision(pt_straight) else 0.0
        danger_right = 1.0 if self.is_collision(pt_right) else 0.0
        danger_left = 1.0 if self.is_collision(pt_left) else 0.0

        # Current direction
        dir_up = 1.0 if self.direction == (0, -1) else 0.0
        dir_right = 1.0 if self.direction == (1, 0) else 0.0
        dir_down = 1.0 if self.direction == (0, 1) else 0.0
        dir_left = 1.0 if self.direction == (-1, 0) else 0.0

        # Relative food position
        food_up = 1.0 if fy < hy else 0.0
        food_right = 1.0 if fx > hx else 0.0
        food_down = 1.0 if fy > hy else 0.0
        food_left = 1.0 if fx < hx else 0.0

        # Normalized Euclidean distance to food
        max_dist = math.hypot(self.width, self.height)
        dist_to_food = math.hypot(fx - hx, fy - hy) / max_dist

        return [
            danger_straight,
            danger_right,
            danger_left,
            dir_up,
            dir_right,
            dir_down,
            dir_left,
            food_up,
            food_right,
            food_down,
            food_left,
            dist_to_food
        ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
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
        ]

    @classmethod
    def get_action_names(cls) -> List[str]:
        return ["Go Straight", "Turn Right", "Turn Left"]

    def step(self, action: int = 0, reward_weights: Dict[str, float] = None) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        if reward_weights is None:
            reward_weights = {
                "food": 10.0,
                "death": -15.0,
                "closer": 0.15,
                "farther": -0.2,
                "step": -0.01
            }

        if self.is_game_over:
            return self.get_state(), 0.0, True, {"score": self.score}

        self.steps += 1
        self.steps_without_food += 1

        headings = self.get_heading_vectors()
        if action == 0:
            self.direction = headings["straight"]
        elif action == 1:
            self.direction = headings["right"]
        elif action == 2:
            self.direction = headings["left"]

        prev_dist = math.hypot(self.food[0] - self.head[0], self.food[1] - self.head[1])
        new_head = (self.head[0] + self.direction[0], self.head[1] + self.direction[1])

        reward = reward_weights.get("step", -0.01)
        done = False

        # Collision check
        if self.is_collision(new_head):
            self.is_game_over = True
            done = True
            reward = reward_weights.get("death", -15.0)
            return self.get_state(), reward, done, {"score": self.score, "reason": "collision"}

        # Hunger limit check
        hunger_limit = self.max_steps_without_food + len(self.body) * 6
        if self.steps_without_food > hunger_limit:
            self.is_game_over = True
            done = True
            reward = reward_weights.get("death", -15.0)
            return self.get_state(), reward, done, {"score": self.score, "reason": "starvation"}

        # Move snake
        self.head = new_head
        self.body.insert(0, new_head)

        # Check food eaten
        if new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            reward = reward_weights.get("food", 10.0)
            self.place_food()
        else:
            self.body.pop()
            new_dist = math.hypot(self.food[0] - self.head[0], self.food[1] - self.head[1])
            if new_dist < prev_dist:
                reward += reward_weights.get("closer", 0.15)
            else:
                reward += reward_weights.get("farther", -0.2)

        return self.get_state(), reward, done, {"score": self.score}
