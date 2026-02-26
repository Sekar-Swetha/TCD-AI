# src/solvers/common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Coord = Tuple[int, int]


@dataclass
class SolveResult:
    path: List[Coord]
    visited_order: List[Coord]          # expansion order (for animation)
    parents: Dict[Coord, Optional[Coord]]  # for debugging/visualization


def reconstruct_path(parents: Dict[Coord, Optional[Coord]], start: Coord, goal: Coord) -> List[Coord]:
    if goal not in parents:
        return []
    cur: Optional[Coord] = goal
    out: List[Coord] = []
    while cur is not None:
        out.append(cur)
        cur = parents.get(cur)
    out.reverse()
    if out and out[0] == start:
        return out
    return []