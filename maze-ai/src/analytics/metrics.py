# src/analytics/metrics.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class RunMetrics:
    # ---------------------------
    # Experiment Identifiers
    # ---------------------------
    algorithm: str
    maze_rows: int
    maze_cols: int
    random_seed: int

    # ---------------------------
    # General Performance Metrics
    # ---------------------------
    solved: bool = False
    execution_time_ms: float = 0.0
    peak_memory_kb: Optional[int] = None

    solution_path_length: int = 0
    solution_cost: float = 0.0
    path_cost_ratio: Optional[float] = None  # cost per step (cost/len)

    # ---------------------------
    # Search Metrics
    # ---------------------------
    states_expanded: int = 0
    states_generated: int = 0
    unique_states_visited: int = 0

    maximum_frontier_size: int = 0
    average_frontier_size: Optional[float] = None
    estimated_branching_factor: Optional[float] = None

    heuristic_type: Optional[str] = None
    heuristic_evaluations: int = 0
    repeated_state_updates: int = 0  # A*: improved g for already-seen state

    # Internal tracking for averages
    _frontier_total: int = 0
    _frontier_samples: int = 0

    # ---------------------------
    # MDP Metrics
    # ---------------------------
    discount_factor: Optional[float] = None
    convergence_threshold: Optional[float] = None

    step_reward: Optional[float] = None
    goal_reward: Optional[float] = None
    wall_penalty: Optional[float] = None

    total_bellman_updates: Optional[int] = None
    final_convergence_error: Optional[float] = None
    policy_reaches_goal: Optional[bool] = None

    value_iteration_steps: Optional[int] = None
    policy_iteration_steps: Optional[int] = None
    policy_evaluation_steps: Optional[int] = None  # total evaluation iterations

    notes: Optional[str] = None

    # ---------------------------
    # Helpers
    # ---------------------------
    def record_frontier(self, size: int) -> None:
        self.maximum_frontier_size = max(self.maximum_frontier_size, size)
        self._frontier_total += size
        self._frontier_samples += 1

    def finalize(self) -> None:
        # avg frontier
        if self._frontier_samples > 0:
            self.average_frontier_size = self._frontier_total / float(self._frontier_samples)

        # branching factor estimate
        if self.states_expanded > 0:
            self.estimated_branching_factor = self.states_generated / float(self.states_expanded)

        # cost/step
        if self.solution_path_length > 0:
            self.path_cost_ratio = self.solution_cost / float(self.solution_path_length)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("_frontier_total", None)
        row.pop("_frontier_samples", None)
        return row