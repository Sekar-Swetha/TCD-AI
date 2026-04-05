import csv
import os
from datetime import datetime

RESULTS_DIR = 'results'


def save_tournament_csv(rows, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fieldnames = ['game', 'agent1', 'agent2', 'n_games',
                  'agent1_wins', 'agent2_wins', 'draws',
                  'agent1_win_pct', 'agent2_win_pct', 'draw_pct', 'timestamp']
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved: {path}  ({len(rows)} rows)")


def tournament_to_row(game_name, agent1_name, agent2_name, results):
    t = results['total']
    w1 = results['agent1_wins']
    w2 = results['agent2_wins']
    d  = results['draws']
    return {
        'game':           game_name,
        'agent1':         agent1_name,
        'agent2':         agent2_name,
        'n_games':        t,
        'agent1_wins':    w1,
        'agent2_wins':    w2,
        'draws':          d,
        'agent1_win_pct': round(w1 / t * 100, 2),
        'agent2_win_pct': round(w2 / t * 100, 2),
        'draw_pct':       round(d  / t * 100, 2),
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def save_training_csv(episodes, results_list, agent_name, game_name, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'outcome'])
        for i, r in enumerate(results_list):
            writer.writerow([i + 1, r])
    print(f"  Training CSV saved: {path}  ({len(results_list)} episodes)")


def save_minimax_timing_csv(data_rows, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fieldnames = ['game', 'variant', 'depth_or_note', 'nodes_visited', 'time_s', 'nodes_per_sec']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
    print(f"  Timing CSV saved: {path}")


def save_hyperparameter_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Hyperparameter CSV saved: {path}")
