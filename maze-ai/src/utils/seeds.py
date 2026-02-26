# src/utils/seeds.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SeedPlan:
    """
    Helper to create repeatable seed lists for experiments.
    """
    base_seed: int = 0

    def seeds(self, k: int) -> List[int]:
        return [self.base_seed + i for i in range(k)]


def parse_seeds(seed: Optional[int], seeds: Optional[Iterable[int]], k: int = 1) -> List[int]:
    """
    Utility for CLI:
      - if seeds provided, use them
      - else if seed provided, return [seed]
      - else return k seeds starting at 0
    """
    if seeds is not None:
        return list(seeds)
    if seed is not None:
        return [seed]
    return SeedPlan(0).seeds(k)