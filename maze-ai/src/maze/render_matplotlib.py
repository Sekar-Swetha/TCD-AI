# src/maze/render_matplotlib.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple
from matplotlib.patches import Patch

import matplotlib.pyplot as plt

from .generator import Maze, N, E, S, W

Coord = Tuple[int, int]


@dataclass
class SaveStyle:
    visited_color: str = "#93c5fd"   # light blue
    path_color: str = "#22c55e"      # green
    start_color: str = "#86efac"     # mint
    goal_color: str = "#fdba74"      # orange
    wall_color: str = "black"
    bg_color: str = "white"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_maze_png(
    maze: Maze,
    out_path: str,
    visited: Optional[Iterable[Coord]] = None,
    path: Optional[Iterable[Coord]] = None,
    title: Optional[str] = None,
    style: SaveStyle = SaveStyle(),
    dpi: int = 200,
) -> None:
    """
    Saves a single PNG showing:
      - maze walls
      - visited nodes (optional)
      - final path (optional)
      - start/goal markers
    """
    visited_set: Set[Coord] = set(visited) if visited is not None else set()
    path_set: Set[Coord] = set(path) if path is not None else set()

    rows, cols = maze.rows, maze.cols

    fig_w = max(4, cols / 6)
    fig_h = max(4, rows / 6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor(style.bg_color)

    # Draw visited and path as filled squares
    def fill_cell(r: int, c: int, color: str, alpha: float = 1.0):
        ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor="none", alpha=alpha))

    for (r, c) in visited_set:
        if (r, c) != maze.start and (r, c) != maze.goal:
            fill_cell(r, c, style.visited_color, alpha=0.8)

    for (r, c) in path_set:
        if (r, c) != maze.start and (r, c) != maze.goal:
            fill_cell(r, c, style.path_color, alpha=0.95)

    # Start/Goal
    sr, sc = maze.start
    gr, gc = maze.goal
    fill_cell(sr, sc, style.start_color, alpha=1.0)
    fill_cell(gr, gc, style.goal_color, alpha=1.0)

    # Draw walls
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c, rows - 1 - r
            x2, y2 = c + 1, rows - r
            mask = maze.walls[r][c]
            if mask & N:
                ax.plot([x1, x2], [y2, y2], color=style.wall_color, linewidth=1.5)
            if mask & E:
                ax.plot([x2, x2], [y1, y2], color=style.wall_color, linewidth=1.5)
            if mask & S:
                ax.plot([x1, x2], [y1, y1], color=style.wall_color, linewidth=1.5)
            if mask & W:
                ax.plot([x1, x1], [y1, y2], color=style.wall_color, linewidth=1.5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title)

    _ensure_dir(os.path.dirname(out_path))

    legend_elements = [
        Patch(facecolor=style.start_color, label="Start"),
        Patch(facecolor=style.goal_color, label="Goal"),
        Patch(facecolor=style.visited_color, label="Visited Nodes"),
        Patch(facecolor=style.path_color, label="Final Path"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        frameon=True
    )
    
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_progress_frames(
    maze: Maze,
    frames_dir: str,
    visited_order: List[Coord],
    path: List[Coord],
    algo: str,
    rows: int,
    cols: int,
    seed: int,
    every: int = 25,
    max_frames: int = 200,
    dpi: int = 160,
) -> None:
    """
    Saves a sequence of PNG frames as the solver explores.
    - every: save a frame every N expansions
    - max_frames: cap to avoid exploding file count
    """
    _ensure_dir(frames_dir)

    total = len(visited_order)
    if total == 0:
        return

    step_indices = list(range(0, total, every))
    if len(step_indices) > max_frames:
        # downsample
        stride = max(1, len(step_indices) // max_frames)
        step_indices = step_indices[::stride]

    for idx, i in enumerate(step_indices):
        visited = visited_order[: i + 1]
        out = os.path.join(
            frames_dir,
            f"{rows}x{cols}_seed{seed}_{algo}_frame{idx:04d}.png",
        )
        save_maze_png(
            maze=maze,
            out_path=out,
            visited=visited,
            path=None,  # during search, show exploration only
            title=f"{algo} | explored {i+1}/{total}",
            dpi=dpi,
        )

    # also save a final "solution overlay" frame
    out_final = os.path.join(
        frames_dir,
        f"{rows}x{cols}_seed{seed}_{algo}_FINAL.png",
    )
    save_maze_png(
        maze=maze,
        out_path=out_final,
        visited=visited_order,
        path=path,
        title=f"{algo} | FINAL",
        dpi=dpi,
    )