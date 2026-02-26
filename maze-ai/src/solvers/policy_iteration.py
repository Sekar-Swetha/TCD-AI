# src/solvers/policy_iteration.py
from __future__ import annotations
from typing import Dict, List, Tuple

from maze.env import MazeEnv
from analytics.metrics import RunMetrics
from .common import SolveResult

Coord = Tuple[int, int]


def _policy_evaluation(env: MazeEnv, policy: Dict[Coord, str], theta: float, max_eval_iters: int) -> Tuple[Dict[Coord, float], float, int]:
    V: Dict[Coord, float] = {s: 0.0 for s in env.states()}
    delta = 0.0
    it = 0
    for it in range(1, max_eval_iters + 1):
        delta = 0.0
        for s in env.states():
            if env.is_goal(s):
                continue
            v_old = V[s]
            a = policy[s]
            V[s] = env.expected_return(V, s, a)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V, float(delta), it


def _policy_improvement(env: MazeEnv, V: Dict[Coord, float], policy: Dict[Coord, str]) -> bool:
    stable = True
    for s in env.states():
        if env.is_goal(s):
            continue
        old = policy[s]
        best_a = old
        best_q = float("-inf")
        for a in env.ACTIONS:
            q = env.expected_return(V, s, a)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
        if best_a != old:
            stable = False
    return stable


def _follow_policy(env: MazeEnv, policy: Dict[Coord, str], max_steps: int) -> List[Coord]:
    s = env.maze.start
    path = [s]
    for _ in range(max_steps):
        if env.is_goal(s):
            break
        a = policy.get(s, "U")
        s2 = env.transitions(s, a)[0][0]
        path.append(s2)
        if s2 == s:
            break
        s = s2
    return path


def solve_policy_iteration(
    env: MazeEnv,
    metrics: RunMetrics,
    theta: float = 1e-6,
    max_policy_iters: int = 10_000,
    max_eval_iters: int = 50_000
) -> SolveResult:
    """
    Policy Iteration:
      - initialize arbitrary policy
      - policy evaluation (iterative)
      - policy improvement
    """
    metrics.discount_factor = env.gamma
    metrics.convergence_threshold = theta
    metrics.step_reward = env.step_reward
    metrics.goal_reward = env.goal_reward
    metrics.wall_penalty = env.wall_reward

    policy: Dict[Coord, str] = {}
    for s in env.states():
        policy[s] = "R"  # simple initial policy

    visited_order: List[Coord] = []  # for optional GUI "activity"

    final_delta = 0.0
    total_eval_sweeps = 0

    for pi_it in range(1, max_policy_iters + 1):
        V, delta, eval_iters = _policy_evaluation(env, policy, theta=theta, max_eval_iters=max_eval_iters)
        final_delta = delta
        total_eval_sweeps += eval_iters

        # record a sweep worth of states for GUI (lightweight)
        for s in env.states():
            visited_order.append(s)

        stable = _policy_improvement(env, V, policy)
        if stable:
            metrics.policy_iteration_steps = pi_it
            metrics.final_convergence_error = float(final_delta)
            metrics.policy_evaluation_steps = total_eval_sweeps
            break

    path = _follow_policy(env, policy, max_steps=env.maze.rows * env.maze.cols * 4)
    solved = (len(path) > 0 and path[-1] == env.maze.goal)

    metrics.solved = solved
    metrics.solution_path_length = max(0, len(path) - 1) if solved else 0
    metrics.solution_cost = float(metrics.solution_path_length)

    return SolveResult(path=path if solved else [], visited_order=visited_order, parents={})