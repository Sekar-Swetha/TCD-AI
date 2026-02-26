# src/solvers/astar.py
from __future__ import annotations
import math
from typing import Dict, Optional, Set, Tuple

from maze.env import MazeEnv
from analytics.metrics import RunMetrics
from utils.priority_queue import PriorityQueue
from .common import SolveResult, reconstruct_path

Coord = Tuple[int, int]


def _h(a: Coord, b: Coord, kind: str, metrics: RunMetrics) -> float:
    (r1, c1), (r2, c2) = a, b
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    metrics.heuristic_evaluations += 1
    if kind == "manhattan":
        return float(dr + dc)
    if kind == "euclidean":
        return math.sqrt(dr * dr + dc * dc)
    raise ValueError("heuristic must be 'manhattan' or 'euclidean'")


def solve_astar(env: MazeEnv, metrics: RunMetrics, heuristic: str) -> SolveResult:
    start = env.maze.start
    goal = env.maze.goal

    metrics.heuristic_type = heuristic

    openpq: PriorityQueue[Coord] = PriorityQueue()
    openpq.push(start, priority=_h(start, goal, heuristic, metrics))

    g: Dict[Coord, float] = {start: 0.0}
    parents: Dict[Coord, Optional[Coord]] = {start: None}
    closed: Set[Coord] = set()
    visited_order = []

    metrics.record_frontier(len(openpq))

    while True:
        try:
            s, _ = openpq.pop()
        except IndexError:
            break

        if s in closed:
            continue
        closed.add(s)

        visited_order.append(s)
        metrics.states_expanded += 1

        if s == goal:
            path = reconstruct_path(parents, start, goal)
            metrics.unique_states_visited = len(g)
            metrics.solution_path_length = max(0, len(path) - 1)
            metrics.solution_cost = float(metrics.solution_path_length)
            metrics.solved = True
            return SolveResult(path=path, visited_order=visited_order, parents=parents)

        for nb in env.neighbors(s):
            metrics.states_generated += 1
            tentative = g[s] + env.cost(s, nb)

            if nb in g and tentative < g[nb]:
                metrics.repeated_state_updates += 1

            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                parents[nb] = s
                f = tentative + _h(nb, goal, heuristic, metrics)
                openpq.push(nb, priority=f)

        metrics.record_frontier(len(openpq))

    metrics.unique_states_visited = len(g)
    metrics.solved = False
    return SolveResult(path=[], visited_order=visited_order, parents=parents)