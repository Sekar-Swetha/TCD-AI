# src/utils/paths.py
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputPaths:
    base_dir: str = "outputs"
    results_dir: str = "outputs/results"
    images_dir: str = "outputs/images"
    plots_dir: str = "outputs/plots"
    results_csv: str = "outputs/results/results.csv"

    def ensure(self) -> "OutputPaths":
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        return self


def image_path(algo: str, rows: int, cols: int, seed: int, extra: str = "") -> str:
    """
    Create a consistent image filename for solved maze output.
    """
    safe_algo = algo.replace("/", "_").replace(" ", "_")
    tag = f"{rows}x{cols}_seed{seed}"
    if extra:
        extra = extra.replace("/", "_").replace(" ", "_")
        name = f"{tag}_{safe_algo}_{extra}.png"
    else:
        name = f"{tag}_{safe_algo}.png"
    return os.path.join("outputs", "images", name)


def plot_path(name: str) -> str:
    safe = name.replace("/", "_").replace(" ", "_")
    return os.path.join("outputs", "plots", f"{safe}.png")

def frames_dir(algo: str, rows: int, cols: int, seed: int) -> str:
    safe_algo = algo.replace("/", "_").replace(" ", "_")
    return os.path.join("outputs", "images", "frames", f"{rows}x{cols}_seed{seed}_{safe_algo}")