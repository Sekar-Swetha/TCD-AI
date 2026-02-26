# src/solvers/dfs.py
from __future__ import annotations
from typing import Dict, Optional, Set, Tuple, List

from maze.env import MazeEnv
from analytics.metrics import RunMetrics
from .common import SolveResult, reconstruct_path

Coord = Tuple[int, int]


def solve_dfs(env: MazeEnv, metrics: RunMetrics) -> SolveResult:
    start = env.maze.start
    goal = env.maze.goal

    stack: List[Coord] = [start]
    visited: Set[Coord] = {start}
    parents: Dict[Coord, Optional[Coord]] = {start: None}
    visited_order: List[Coord] = []

    metrics.record_frontier(len(stack))

    while stack:
        s = stack.pop()
        visited_order.append(s)
        metrics.states_expanded += 1

        if s == goal:
            path = reconstruct_path(parents, start, goal)
            metrics.unique_states_visited = len(visited)
            metrics.solution_path_length = max(0, len(path) - 1)
            metrics.solution_cost = float(metrics.solution_path_length)
            metrics.solved = True
            return SolveResult(path=path, visited_order=visited_order, parents=parents)

        nbs = env.neighbors(s)
        for nb in reversed(nbs):
            metrics.states_generated += 1
            if nb in visited:
                continue
            visited.add(nb)
            parents[nb] = s
            stack.append(nb)

        metrics.record_frontier(len(stack))

    metrics.unique_states_visited = len(visited)
    metrics.solved = False
    return SolveResult(path=[], visited_order=visited_order, parents=parents)