# src/utils/priority_queue.py
from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


@dataclass(order=True)
class _PQItem(Generic[T]):
    priority: float
    tie: int
    item: T = field(compare=False)


class PriorityQueue(Generic[T]):
    """
    Min-priority queue with:
      - stable tie-breaker
      - optional decrease-key via pushing new entries and checking best-known priorities
    """

    def __init__(self):
        self._heap: List[_PQItem[T]] = []
        self._tie = 0
        self._best: Dict[T, float] = {}

    def push(self, item: T, priority: float) -> None:
        best = self._best.get(item)
        if best is None or priority < best:
            self._best[item] = priority
            heapq.heappush(self._heap, _PQItem(priority=priority, tie=self._tie, item=item))
            self._tie += 1

    def pop(self) -> Tuple[T, float]:
        while self._heap:
            top = heapq.heappop(self._heap)
            # discard stale entries
            if self._best.get(top.item) == top.priority:
                return top.item, top.priority
        raise IndexError("pop from empty PriorityQueue")

    def empty(self) -> bool:
        return len(self._best) == 0 or all(
            self._best.get(it.item) != it.priority for it in self._heap
        )

    def __len__(self) -> int:
        # approximate current (includes stale in heap, but best dict is true active set)
        return len(self._best)

    def peek_priority(self, item: T) -> Optional[float]:
        return self._best.get(item)