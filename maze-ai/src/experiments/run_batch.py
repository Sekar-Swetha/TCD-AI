# src/experiments/run_batch.py
from __future__ import annotations
import subprocess
import sys

SIZES = [11, 31, 51, 81, 111]
SEEDS = list(range(10))
ALGOS = ["bfs", "dfs", "astar_manhattan", "astar_euclidean", "value", "policy"]

def run():
    for n in SIZES:
        for seed in SEEDS:
            for algo in ALGOS:
                cmd = [sys.executable, "src/main.py", "--rows", str(n), "--cols", str(n),
                       "--seed", str(seed), "--algo", algo]
                print("RUN:", " ".join(cmd))
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run()