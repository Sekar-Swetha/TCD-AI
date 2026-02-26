# src/maze/generator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

Coord = Tuple[int, int]


@dataclass(frozen=True)
class Maze:
    """
    Perfect maze on a grid using 'cell walls' representation.

    walls[r][c] is a 4-bit mask: N=1, E=2, S=4, W=8.
    If a bit is set, that wall exists.
    """
    rows: int
    cols: int
    walls: List[List[int]]
    start: Coord
    goal: Coord


# Wall bit flags
N, E, S, W = 1, 2, 4, 8

DIRS = [
    (-1, 0, N, S),  # move up: remove N from current, S from next
    (0, 1, E, W),   # move right
    (1, 0, S, N),   # move down
    (0, -1, W, E),  # move left
]


def generate_maze(rows: int, cols: int, seed: Optional[int] = None,
                  start: Coord = (0, 0), goal: Optional[Coord] = None) -> Maze:
    """
    Generate a perfect maze using iterative recursive backtracking (DFS carving).
    This is a GENERATOR ONLY — independent of BFS/DFS/A*/MDP solvers.
    """
    if rows < 2 or cols < 2:
        raise ValueError("Maze must be at least 2x2.")
    if goal is None:
        goal = (rows - 1, cols - 1)

    rng = random.Random(seed)

    # Initialize all walls present
    walls = [[N | E | S | W for _ in range(cols)] for _ in range(rows)]
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    sr, sc = start
    stack = [(sr, sc)]
    visited[sr][sc] = True

    while stack:
        r, c = stack[-1]

        # collect unvisited neighbors
        neigh = []
        for dr, dc, w_curr, w_next in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                neigh.append((nr, nc, w_curr, w_next))

        if not neigh:
            stack.pop()
            continue

        nr, nc, w_curr, w_next = rng.choice(neigh)

        # remove walls between current and next
        walls[r][c] &= ~w_curr
        walls[nr][nc] &= ~w_next

        visited[nr][nc] = True
        stack.append((nr, nc))

    return Maze(rows=rows, cols=cols, walls=walls, start=start, goal=goal)


def has_wall(maze: Maze, r: int, c: int, direction_bit: int) -> bool:
    return (maze.walls[r][c] & direction_bit) != 0