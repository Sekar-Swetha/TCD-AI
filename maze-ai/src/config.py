# src/config.py
from __future__ import annotations
from dataclasses import dataclass
from utils.paths import OutputPaths


@dataclass
class Config:
    """
    Central configuration for the project.

    Controls:
      - MDP parameters
      - Animation settings
      - Output directories
    """

    # ---------------------------
    # MDP Parameters
    # ---------------------------
    gamma: float = 0.99
    theta: float = 1e-6
    slip_prob: float = 0.0  # 0 = deterministic

    step_reward: float = -0.01
    goal_reward: float = 1.0
    wall_reward: float = -0.05

    # ---------------------------
    # GUI / Animation
    # ---------------------------
    anim_delay_ms: int = 10
    cell_px: int = 24

    # ---------------------------
    # Output Paths
    # ---------------------------
    paths: OutputPaths = OutputPaths().ensure()

    @property
    def results_csv(self) -> str:
        return self.paths.results_csv

    @property
    def images_dir(self) -> str:
        return self.paths.images_dir

    @property
    def plots_dir(self) -> str:
        return self.paths.plots_dir