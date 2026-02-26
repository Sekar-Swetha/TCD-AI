# src/maze/env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .generator import Maze, N, E, S, W, has_wall

Coord = Tuple[int, int]


@dataclass(frozen=True)
class MazeEnv:
    """
    Environment wrapper around Maze that provides:
      - neighbors() for graph search (DFS/BFS/A*)
      - MDP-style actions + transitions + rewards for Value/Policy iteration
    """
    maze: Maze
    step_reward: float = -0.01
    goal_reward: float = 1.0
    wall_reward: float = -0.05  # used if you choose to "bounce" on walls
    gamma: float = 0.99
    slip_prob: float = 0.0      # 0.0 = deterministic; set >0 for stochastic actions (optional)

    ACTIONS: Tuple[str, ...] = ("U", "R", "D", "L")

    def in_bounds(self, s: Coord) -> bool:
        r, c = s
        return 0 <= r < self.maze.rows and 0 <= c < self.maze.cols

    def is_goal(self, s: Coord) -> bool:
        return s == self.maze.goal

    # ----------------------------
    # Search helpers (DFS/BFS/A*)
    # ----------------------------
    def neighbors(self, s: Coord) -> List[Coord]:
        """Return valid neighbor states (deterministic, no diagonals)."""
        r, c = s
        out: List[Coord] = []

        if not has_wall(self.maze, r, c, N) and r - 1 >= 0:
            out.append((r - 1, c))
        if not has_wall(self.maze, r, c, E) and c + 1 < self.maze.cols:
            out.append((r, c + 1))
        if not has_wall(self.maze, r, c, S) and r + 1 < self.maze.rows:
            out.append((r + 1, c))
        if not has_wall(self.maze, r, c, W) and c - 1 >= 0:
            out.append((r, c - 1))

        return out

    def cost(self, s: Coord, s2: Coord) -> float:
        """Uniform cost for search algorithms (unweighted graph)."""
        return 1.0

    # ----------------------------
    # MDP helpers (Value/Policy)
    # ----------------------------
    def states(self) -> Iterable[Coord]:
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                yield (r, c)

    def _move(self, s: Coord, a: str) -> Coord:
        """Deterministic move: if blocked by wall, you stay in place."""
        r, c = s
        if a == "U":
            if has_wall(self.maze, r, c, N) or r - 1 < 0:
                return s
            return (r - 1, c)
        if a == "R":
            if has_wall(self.maze, r, c, E) or c + 1 >= self.maze.cols:
                return s
            return (r, c + 1)
        if a == "D":
            if has_wall(self.maze, r, c, S) or r + 1 >= self.maze.rows:
                return s
            return (r + 1, c)
        if a == "L":
            if has_wall(self.maze, r, c, W) or c - 1 < 0:
                return s
            return (r, c - 1)
        raise ValueError(f"Unknown action: {a}")

    def reward(self, s: Coord, a: str, s2: Coord) -> float:
        """Reward function for MDP."""
        if self.is_goal(s):
            # terminal state (common convention)
            return 0.0
        if self.is_goal(s2):
            return self.goal_reward
        if s2 == s and a in self.ACTIONS:
            # bounced into wall
            return self.wall_reward
        return self.step_reward

    def transitions(self, s: Coord, a: str) -> List[Tuple[Coord, float]]:
        """
        Return list of (next_state, probability).
        If slip_prob > 0, action may slip to left/right (relative) with small probability.
        """
        if self.is_goal(s):
            return [(s, 1.0)]

        if self.slip_prob <= 0.0:
            return [(self._move(s, a), 1.0)]

        # Optional stochasticity: intended action with 1-slip, plus two side actions split.
        # U side actions: L/R ; R side actions: U/D ; D side actions: L/R ; L side actions: U/D
        side = {
            "U": ("L", "R"),
            "R": ("U", "D"),
            "D": ("L", "R"),
            "L": ("U", "D"),
        }[a]

        p_main = 1.0 - self.slip_prob
        p_side = self.slip_prob / 2.0

        ns_main = self._move(s, a)
        ns_s1 = self._move(s, side[0])
        ns_s2 = self._move(s, side[1])

        # combine probabilities if two outcomes land on same state
        probs: Dict[Coord, float] = {}
        probs[ns_main] = probs.get(ns_main, 0.0) + p_main
        probs[ns_s1] = probs.get(ns_s1, 0.0) + p_side
        probs[ns_s2] = probs.get(ns_s2, 0.0) + p_side

        return list(probs.items())

    def expected_return(self, V: Dict[Coord, float], s: Coord, a: str) -> float:
        """One-step Bellman backup for Q(s,a) from V."""
        total = 0.0
        for s2, p in self.transitions(s, a):
            r = self.reward(s, a, s2)
            total += p * (r + self.gamma * V[s2])
        return total