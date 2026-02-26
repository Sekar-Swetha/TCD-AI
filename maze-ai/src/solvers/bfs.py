# src/solvers/bfs.py
from __future__ import annotations
from collections import deque
from typing import Dict, Optional, Set, Tuple

from maze.env import MazeEnv
from analytics.metrics import RunMetrics
from .common import SolveResult, reconstruct_path

Coord = Tuple[int, int]


def solve_bfs(env: MazeEnv, metrics: RunMetrics) -> SolveResult:
    start = env.maze.start
    goal = env.maze.goal

    q = deque([start])
    visited: Set[Coord] = {start}
    parents: Dict[Coord, Optional[Coord]] = {start: None}
    visited_order = []

    metrics.record_frontier(len(q))

    while q:
        s = q.popleft()
        visited_order.append(s)
        metrics.states_expanded += 1

        if s == goal:
            path = reconstruct_path(parents, start, goal)
            metrics.unique_states_visited = len(visited)
            metrics.solution_path_length = max(0, len(path) - 1)
            metrics.solution_cost = float(metrics.solution_path_length)
            metrics.solved = True
            return SolveResult(path=path, visited_order=visited_order, parents=parents)

        for nb in env.neighbors(s):
            metrics.states_generated += 1
            if nb in visited:
                continue
            visited.add(nb)
            parents[nb] = s
            q.append(nb)

        metrics.record_frontier(len(q))

    metrics.unique_states_visited = len(visited)
    metrics.solved = False
    return SolveResult(path=[], visited_order=visited_order, parents=parents)