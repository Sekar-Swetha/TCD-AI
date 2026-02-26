# src/main.py
from __future__ import annotations

import argparse
import os
import sys
import time
import resource

# Ensure imports work when running: python src/main.py ...
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from config import Config
from analytics.metrics import RunMetrics
from analytics.logger import CSVLogger

from maze.generator import generate_maze
from maze.env import MazeEnv
#from maze.render_tk import MazeTkRenderer

from solvers.bfs import solve_bfs
from solvers.dfs import solve_dfs
from solvers.astar import solve_astar
from solvers.value_iteration import solve_value_iteration
from solvers.policy_iteration import solve_policy_iteration
from utils.paths import frames_dir, image_path
from maze.render_matplotlib import save_maze_png, save_progress_frames

def parse_args():
    p = argparse.ArgumentParser(description="CS7IS2 Maze Solver + Analytics")

    # Maze params
    p.add_argument("--rows", type=int, default=11)
    p.add_argument("--cols", type=int, default=11)
    p.add_argument("--seed", type=int, default=0)

    # Algorithm choice
    p.add_argument(
        "--algo",
        type=str,
        default="bfs",
        choices=["bfs", "dfs", "astar_manhattan", "astar_euclidean", "value", "policy"],
    )

    # MDP params (override config defaults if provided)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--theta", type=float, default=None)
    p.add_argument("--slip", type=float, default=None)

    # GUI params
    p.add_argument("--cell_px", type=int, default=None)
    p.add_argument("--anim_delay_ms", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()  # creates output folders via OutputPaths().ensure()

    # Override config if CLI flags provided
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.theta is not None:
        cfg.theta = args.theta
    if args.slip is not None:
        cfg.slip_prob = args.slip

    if args.cell_px is not None:
        cfg.cell_px = args.cell_px
    if args.anim_delay_ms is not None:
        cfg.anim_delay_ms = args.anim_delay_ms

    # ---- generate maze (independent of solver) ----
    maze = generate_maze(rows=args.rows, cols=args.cols, seed=args.seed)

    # ---- build env ----
    env = MazeEnv(
        maze=maze,
        gamma=cfg.gamma,
        slip_prob=cfg.slip_prob,
        step_reward=cfg.step_reward,
        goal_reward=cfg.goal_reward,
        wall_reward=cfg.wall_reward,
    )

    # ---- init metrics ----
    metrics = RunMetrics(
        algorithm=args.algo,
        maze_rows=args.rows,
        maze_cols=args.cols,
        random_seed=args.seed,
    )

    # ---- run solver ----
    t0 = time.perf_counter()

    if args.algo == "bfs":
        result = solve_bfs(env, metrics)
    elif args.algo == "dfs":
        result = solve_dfs(env, metrics)
    elif args.algo == "astar_manhattan":
        result = solve_astar(env, metrics, heuristic="manhattan")
    elif args.algo == "astar_euclidean":
        result = solve_astar(env, metrics, heuristic="euclidean")
    elif args.algo == "value":
        result = solve_value_iteration(env, metrics, theta=cfg.theta)
    elif args.algo == "policy":
        result = solve_policy_iteration(env, metrics, theta=cfg.theta)
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    t1 = time.perf_counter()
    metrics.execution_time_ms = (t1 - t0) * 1000.0
    metrics.peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ---- print analytics ----
    print("\n=== RUN SUMMARY ===")
    print(f"Algo: {metrics.algorithm}")
    print(f"Maze: {metrics.maze_rows}x{metrics.maze_cols} (seed={metrics.random_seed})")
    print(f"Solved: {metrics.solved}")
    print(f"Runtime: {metrics.execution_time_ms:.3f} ms")

    if metrics.heuristic_type:
        print(f"Heuristic: {metrics.heuristic_type}")

    if args.algo in ("bfs", "dfs", "astar_manhattan", "astar_euclidean"):
        print(f"States expanded: {metrics.states_expanded}")
        print(f"Unique visited:  {metrics.unique_states_visited}")
        print(f"Max frontier:    {metrics.maximum_frontier_size}")
        print(f"Path length:     {metrics.solution_path_length}")
        print(f"Solution cost:   {metrics.solution_cost}")
    else:
        print(f"MDP gamma:       {metrics.discount_factor}")
        print(f"MDP theta:       {metrics.convergence_threshold}")
        print(f"Step reward:     {metrics.step_reward}")
        print(f"Goal reward:     {metrics.goal_reward}")
        print(f"Wall penalty:    {metrics.wall_penalty}")
        # Iterations / convergence info
        iters = metrics.value_iteration_steps or metrics.policy_iteration_steps
        print(f"Iterations:      {iters}")
        print(f"Final delta:     {metrics.final_convergence_error}")
        print(f"Path length:     {metrics.solution_path_length}")
        print(f"Peak memory:     {metrics.peak_memory_kb} KB")

    metrics.finalize()
    # ---- log CSV ----
    logger = CSVLogger(cfg.results_csv)
    logger.log(metrics)
    print(f"\nSaved metrics -> {cfg.results_csv}")

    title = f"{metrics.algorithm} | {metrics.maze_rows}x{metrics.maze_cols} | seed={metrics.random_seed}"

    # ---- save images for report ----
    # 1) final image (maze + visited + path)
    final_img = image_path(metrics.algorithm, metrics.maze_rows, metrics.maze_cols, metrics.random_seed, extra="final")
    save_maze_png(
        maze=maze,
        out_path=final_img,
        visited=result.visited_order,
        path=result.path,
        title=title,
        dpi=220,
    )
    print(f"Saved final image -> {final_img}")

    # 2) progress frames (optional but great for report/demo)
    fd = frames_dir(metrics.algorithm, metrics.maze_rows, metrics.maze_cols, metrics.random_seed)
    save_progress_frames(
        maze=maze,
        frames_dir=fd,
        visited_order=result.visited_order,
        path=result.path,
        algo=metrics.algorithm,
        rows=metrics.maze_rows,
        cols=metrics.maze_cols,
        seed=metrics.random_seed,
        every=25,        # tune: smaller = more frames
        max_frames=180,  # avoid huge folders
        dpi=160,
    )
    print(f"Saved progress frames -> {fd}")

    # ---- GUI animation ----
    
    # renderer = MazeTkRenderer(
    #     maze,
    #     cell_px=cfg.cell_px,
    #     anim_delay_ms=cfg.anim_delay_ms,
    #     title=title,
    # )
    # renderer.animate(result.visited_order, result.path)
    # renderer.run()


if __name__ == "__main__":
    main()