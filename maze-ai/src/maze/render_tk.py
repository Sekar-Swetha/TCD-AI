# src/maze/render_tk.py
from __future__ import annotations
import tkinter as tk
from typing import List, Tuple, Optional

from .generator import Maze, N, E, S, W

Coord = Tuple[int, int]


class MazeTkRenderer:
    """
    Minimal Tkinter renderer + animation:
      - draws maze walls
      - animates visited_order (search expansion)
      - then animates final path
    """

    def __init__(
        self,
        maze: Maze,
        cell_px: int = 24,
        anim_delay_ms: int = 10,
        title: str = "CS7IS2 Maze Solver"
    ):
        self.maze = maze
        self.cell_px = cell_px
        self.anim_delay_ms = anim_delay_ms

        w = maze.cols * cell_px + 2
        h = maze.rows * cell_px + 2

        self.root = tk.Tk()
        self.root.title(title)
        self.canvas = tk.Canvas(self.root, width=w, height=h)
        self.canvas.pack()

        self._rects = [[None for _ in range(maze.cols)] for _ in range(maze.rows)]
        self._draw_grid()
        self._draw_walls()
        self._color_cell(maze.start, outline="black", fill="#c7f9cc")
        self._color_cell(maze.goal, outline="black", fill="#ffd6a5")

    def _cell_bbox(self, r: int, c: int):
        x1 = c * self.cell_px + 1
        y1 = r * self.cell_px + 1
        x2 = x1 + self.cell_px
        y2 = y1 + self.cell_px
        return x1, y1, x2, y2

    def _draw_grid(self):
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                x1, y1, x2, y2 = self._cell_bbox(r, c)
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#e5e7eb", fill="white")
                self._rects[r][c] = rect

    def _draw_walls(self):
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                x1, y1, x2, y2 = self._cell_bbox(r, c)
                mask = self.maze.walls[r][c]

                if mask & N:
                    self.canvas.create_line(x1, y1, x2, y1, width=2)
                if mask & E:
                    self.canvas.create_line(x2, y1, x2, y2, width=2)
                if mask & S:
                    self.canvas.create_line(x1, y2, x2, y2, width=2)
                if mask & W:
                    self.canvas.create_line(x1, y1, x1, y2, width=2)

    def _color_cell(self, s: Coord, outline: str = "#e5e7eb", fill: str = "white"):
        r, c = s
        rect = self._rects[r][c]
        self.canvas.itemconfig(rect, outline=outline, fill=fill)

    def animate(self, visited_order: List[Coord], path: List[Coord], on_done: Optional[callable] = None):
        """
        visited_order: expansion order (can be long). We'll animate lightly.
        path: final solution path (start->goal).
        """
        # compress visited animation a bit for speed (skip factor)
        skip = 1
        if len(visited_order) > 4000:
            skip = 10
        elif len(visited_order) > 2000:
            skip = 5
        elif len(visited_order) > 1000:
            skip = 2

        visited = visited_order[::skip]

        def step_visit(i: int):
            if i >= len(visited):
                step_path(0)
                return
            s = visited[i]
            if s != self.maze.start and s != self.maze.goal:
                self._color_cell(s, fill="#dbeafe")  # light blue
            self.root.after(self.anim_delay_ms, lambda: step_visit(i + 1))

        def step_path(i: int):
            if i >= len(path):
                if on_done:
                    on_done()
                return
            s = path[i]
            if s != self.maze.start and s != self.maze.goal:
                self._color_cell(s, fill="#86efac")  # green
            self.root.after(max(5, self.anim_delay_ms), lambda: step_path(i + 1))

        step_visit(0)

    def run(self):
        self.root.mainloop()