# src/solvers/value_iteration.py
from __future__ import annotations
import time
from typing import Dict, List, Tuple

from maze.env import MazeEnv
from analytics.metrics import RunMetrics
from .common import SolveResult

Coord = Tuple[int, int]


def _derive_policy(env: MazeEnv, V: Dict[Coord, float]) -> Dict[Coord, str]:
    policy: Dict[Coord, str] = {}
    for s in env.states():
        if env.is_goal(s):
            policy[s] = "U"  # arbitrary; terminal
            continue
        best_a = None
        best_q = float("-inf")
        for a in env.ACTIONS:
            q = env.expected_return(V, s, a)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a if best_a is not None else "U"
    return policy


def _follow_policy(env: MazeEnv, policy: Dict[Coord, str], max_steps: int) -> List[Coord]:
    s = env.maze.start
    path = [s]
    for _ in range(max_steps):
        if env.is_goal(s):
            break
        a = policy.get(s, "U")
        s2 = env.transitions(s, a)[0][0]  # deterministic or most-probable first
        path.append(s2)
        if s2 == s:
            # stuck bouncing
            break
        s = s2
    return path


def solve_value_iteration(env: MazeEnv, metrics: RunMetrics, theta: float = 1e-6, max_iters: int = 200_000) -> SolveResult:
    """
    Standard Value Iteration for an MDP maze.
    Returns a policy-induced path from start to goal (if it reaches goal).
    """
    metrics.discount_factor = env.gamma
    metrics.convergence_threshold = theta
    metrics.step_reward = env.step_reward
    metrics.goal_reward = env.goal_reward
    metrics.wall_penalty = env.wall_reward

    V: Dict[Coord, float] = {s: 0.0 for s in env.states()}

    visited_order: List[Coord] = []  # for GUI: we'll show "sweeps" as visits (optional)

    it = 0
    delta = 0.0
    for it in range(1, max_iters + 1):
        delta = 0.0
        for s in env.states():
            if env.is_goal(s):
                continue
            v_old = V[s]
            best = float("-inf")
            for a in env.ACTIONS:
                best = max(best, env.expected_return(V, s, a))
            V[s] = best
            delta = max(delta, abs(v_old - V[s]))

            # for a tiny bit of visual activity (not too heavy)
            visited_order.append(s)

        if delta < theta:
            break

    policy = _derive_policy(env, V)
    path = _follow_policy(env, policy, max_steps=env.maze.rows * env.maze.cols * 4)

    solved = (len(path) > 0 and path[-1] == env.maze.goal)

    metrics.value_iteration_steps = it
    metrics.final_convergence_error = float(delta)
    metrics.solved = solved
    metrics.solution_path_length = max(0, len(path) - 1) if solved else 0
    metrics.solution_cost = float(metrics.solution_path_length)

    # We’ll store policy in parents as None (GUI doesn’t need parents for MDP)
    return SolveResult(path=path if solved else [], visited_order=visited_order, parents={})