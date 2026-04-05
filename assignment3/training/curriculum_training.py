import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.minimax import MinimaxAgent, connect4_eval
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from training.evaluate import run_tournament
from training.results_logger import save_tournament_csv, tournament_to_row, save_training_csv
from visualizer import plot_dqn_loss_curve
import config

RESULTS_TTT = os.path.join('results', 'ttt')
RESULTS_C4  = os.path.join('results', 'connect4')
os.makedirs(RESULTS_TTT, exist_ok=True)
os.makedirs(RESULTS_C4, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)


class RandomAgent:
    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
    def select_action(self, game, legal_moves=None):
        return random.choice(legal_moves or game.get_legal_moves())


def curriculum_train_ql(game_cls, game_name, episodes_per_stage,
                        ql_path, is_c4=False):
    game = game_cls()
    rand_opp    = RandomAgent(player=BaseGame.PLAYER2)
    default_opp = DefaultOpponent(player=BaseGame.PLAYER2)
    mm_opp      = MinimaxAgent(player=BaseGame.PLAYER2, use_alpha_beta=True,
                               depth_limit=config.MINIMAX_DEPTH_C4 if is_c4 else None,
                               eval_fn=connect4_eval if is_c4 else None)

    agent = QLearningAgent(
        player=BaseGame.PLAYER1,
        alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=config.QL_EPSILON_DECAY,
    )

    stages = [
        ('vs Random',      rand_opp,    episodes_per_stage[0]),
        ('vs Default',     default_opp, episodes_per_stage[1]),
        ('vs Minimax+AB',  mm_opp,      episodes_per_stage[2]),
    ]

    all_results = {}
    print(f"\n--- Q-Learning Curriculum ({game_name}) ---")
    for stage_name, opp, eps in stages:
        print(f"  Stage: {stage_name} ({eps} episodes)")
        opp.player = BaseGame.PLAYER2
        results = agent.train(game, opp, num_episodes=eps, verbose=True)
        all_results[stage_name] = results

    agent.save(ql_path)
    return agent, all_results


def curriculum_train_dqn(game_cls, game_name, episodes_per_stage,
                         dqn_path, is_c4=False):
    game = game_cls()
    rand_opp    = RandomAgent(player=BaseGame.PLAYER2)
    default_opp = DefaultOpponent(player=BaseGame.PLAYER2)
    mm_opp      = MinimaxAgent(player=BaseGame.PLAYER2, use_alpha_beta=True,
                               depth_limit=config.MINIMAX_DEPTH_C4 if is_c4 else None,
                               eval_fn=connect4_eval if is_c4 else None)

    in_size  = config.DQN_INPUT_C4  if is_c4 else config.DQN_INPUT_TTT
    out_size = config.DQN_OUTPUT_C4 if is_c4 else config.DQN_OUTPUT_TTT
    hidden   = config.DQN_HIDDEN_C4 if is_c4 else config.DQN_HIDDEN_TTT

    agent = DQNAgent(
        player=BaseGame.PLAYER1,
        input_size=in_size, output_size=out_size, hidden_sizes=hidden,
        lr=config.DQN_LR, gamma=config.DQN_GAMMA,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=config.DQN_EPSILON_DECAY,
        batch_size=config.DQN_BATCH_SIZE, buffer_capacity=config.DQN_BUFFER_CAPACITY,
        target_update_freq=config.DQN_TARGET_UPDATE,
    )

    stages = [
        ('vs Random',      rand_opp,    episodes_per_stage[0]),
        ('vs Default',     default_opp, episodes_per_stage[1]),
        ('vs Minimax+AB',  mm_opp,      episodes_per_stage[2]),
    ]

    all_results = {}
    print(f"\n--- DQN Curriculum ({game_name}) ---")
    for stage_name, opp, eps in stages:
        print(f"  Stage: {stage_name} ({eps} episodes)")
        opp.player = BaseGame.PLAYER2
        results = agent.train(game, opp, num_episodes=eps, verbose=True)
        all_results[stage_name] = results

    agent.save(dqn_path)

    loss_path = os.path.join(
        RESULTS_C4 if is_c4 else RESULTS_TTT,
        f'dqn_loss_curriculum.png'
    )
    plot_dqn_loss_curve(agent.loss_history, f'DQN {game_name} curriculum', path=loss_path)
    print(f"  Saved DQN loss curve: {loss_path}")

    return agent, all_results


def plot_curriculum_curves(all_results_ql, all_results_dqn, title, path, window=300):
    stage_colors = {'vs Random': '#2a9d8f', 'vs Default': '#e9c46a', 'vs Minimax+AB': '#e76f51'}
    kernel = np.ones(window) / window

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (agent_name, all_results) in zip(axes, [('Q-Learning', all_results_ql),
                                                     ('DQN',        all_results_dqn)]):
        offset = 0
        for stage, results in all_results.items():
            wins = np.array([1.0 if r == 'win' else 0.0 for r in results])
            if len(wins) >= window:
                smoothed = np.convolve(wins, kernel, mode='valid')
                x = np.arange(offset + window, offset + len(wins) + 1)
                ax.plot(x, smoothed * 100, color=stage_colors.get(stage, 'gray'),
                        lw=1.5, label=stage)
                ax.axvline(offset, color='black', ls='--', lw=0.8, alpha=0.5)
            offset += len(wins)

        ax.set_xlabel('Cumulative episode')
        ax.set_ylabel('Win rate (%)')
        ax.set_title(f'{agent_name} — Curriculum Training')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved curriculum chart: {path}")


def run_ttt():
    ql_path  = os.path.join(config.MODEL_DIR, 'ttt_ql_curriculum.pkl')
    dqn_path = os.path.join(config.MODEL_DIR, 'ttt_dqn_curriculum.pt')

    eps = [20_000, 20_000, 10_000]

    ql_agent,  ql_res  = curriculum_train_ql(TicTacToe,  'TTT', eps, ql_path,  is_c4=False)
    dqn_agent, dqn_res = curriculum_train_dqn(TicTacToe, 'TTT', eps, dqn_path, is_c4=False)

    plot_curriculum_curves(
        ql_res, dqn_res,
        title='TTT Curriculum Training: Random → Default → Minimax+AB',
        path=os.path.join(RESULTS_TTT, 'curriculum_curves.png'),
    )

    game = TicTacToe()
    default_opp = DefaultOpponent(player=BaseGame.PLAYER2)
    csv_rows = []
    for agent, name in [(ql_agent, 'QL-Curriculum'), (dqn_agent, 'DQN-Curriculum')]:
        r = run_tournament(agent, default_opp, game, n_games=200, swap_players=True)
        csv_rows.append(tournament_to_row('TTT', name, 'Default', r))
        print(f"  {name} vs Default: {r['agent1_wins']/r['total']*100:.1f}% wins")

    save_tournament_csv(csv_rows, os.path.join(RESULTS_TTT, 'curriculum_results.csv'))
    return ql_agent, dqn_agent


def run_c4():
    ql_path  = os.path.join(config.MODEL_DIR, 'c4_ql_curriculum.pkl')
    dqn_path = os.path.join(config.MODEL_DIR, 'c4_dqn_curriculum.pt')

    # more random episodes (state space is huge), fewer minimax (too slow for C4)
    eps = [50_000, 30_000, 5_000]

    ql_agent,  ql_res  = curriculum_train_ql(Connect4,  'C4', eps, ql_path,  is_c4=True)
    dqn_agent, dqn_res = curriculum_train_dqn(Connect4, 'C4', eps, dqn_path, is_c4=True)

    plot_curriculum_curves(
        ql_res, dqn_res,
        title='C4 Curriculum Training: Random → Default → Minimax+AB(d3)',
        path=os.path.join(RESULTS_C4, 'curriculum_curves.png'),
    )

    game = Connect4()
    rand_opp = RandomAgent(player=BaseGame.PLAYER2)
    csv_rows = []
    for agent, name in [(ql_agent, 'QL-Curriculum'), (dqn_agent, 'DQN-Curriculum')]:
        r = run_tournament(agent, rand_opp, game, n_games=100, swap_players=True)
        csv_rows.append(tournament_to_row('C4', name, 'Random', r))
        print(f"  {name} vs Random: {r['agent1_wins']/r['total']*100:.1f}% wins")

    save_tournament_csv(csv_rows, os.path.join(RESULTS_C4, 'curriculum_results.csv'))
    return ql_agent, dqn_agent


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--game', choices=['ttt', 'connect4', 'both'], default='ttt')
    args = p.parse_args()

    if args.game in ('ttt', 'both'):
        run_ttt()
    if args.game in ('connect4', 'both'):
        run_c4()
