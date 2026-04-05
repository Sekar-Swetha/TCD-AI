import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.minimax import MinimaxAgent, connect4_eval
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
import config

RESULTS_TTT = os.path.join('results', 'ttt')
RESULTS_C4  = os.path.join('results', 'connect4')


class RandomAgent:
    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
    def select_action(self, game, legal_moves=None):
        return random.choice(legal_moves or game.get_legal_moves())


def play_fixed(a1, a2, game, n, a1_role):
    a2_role = BaseGame.PLAYER2 if a1_role == BaseGame.PLAYER1 else BaseGame.PLAYER1
    a1.player = a1_role
    a2.player = a2_role
    w1 = w2 = dr = 0
    for _ in range(n):
        game.reset()
        while not game.is_terminal():
            cur = game.get_current_player()
            move = a1.select_action(game) if cur == a1_role else a2.select_action(game)
            game.make_move(move)
        winner = game.check_winner()
        if winner == a1_role:   w1 += 1
        elif winner == a2_role: w2 += 1
        else:                   dr += 1
    return w1, w2, dr


def first_mover_data(a1, a2, game, n=100):
    w1, l1, d1 = play_fixed(a1, a2, game, n, BaseGame.PLAYER1)
    w2, l2, d2 = play_fixed(a1, a2, game, n, BaseGame.PLAYER2)
    p1_win_pct  = w1 / n * 100
    p2_win_pct  = w2 / n * 100
    draw_pct    = (d1 + d2) / (2*n) * 100
    return p1_win_pct, p2_win_pct, draw_pct


def load_ql(path, player=BaseGame.PLAYER1):
    a = QLearningAgent(player=player, alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
                       epsilon=config.QL_EPSILON_MIN, epsilon_min=config.QL_EPSILON_MIN,
                       epsilon_decay=config.QL_EPSILON_DECAY)
    a.load(path)
    return a


def load_dqn(path, in_size, out_size, hidden, player=BaseGame.PLAYER1):
    a = DQNAgent(player=player, input_size=in_size, output_size=out_size, hidden_sizes=hidden,
                 lr=config.DQN_LR, gamma=config.DQN_GAMMA,
                 epsilon=config.DQN_EPSILON_MIN, epsilon_min=config.DQN_EPSILON_MIN,
                 epsilon_decay=config.DQN_EPSILON_DECAY, batch_size=config.DQN_BATCH_SIZE)
    a.load(path)
    return a


def plot_p1p2(labels, p1_wins, p2_wins, draws, title, path):
    x = np.arange(len(labels))
    w = 0.55
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x, p1_wins, w, label='Win as P1 (goes first)', color='#2a9d8f')
    b2 = ax.bar(x, draws,   w, bottom=p1_wins, label='Draw', color='#e9c46a')
    b3 = ax.bar(x, p2_wins, w, bottom=np.array(p1_wins)+np.array(draws),
                label='Win as P2 (goes second)', color='#e76f51')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel('Win rate (%)'); ax.set_ylim(0, 115)
    ax.axhline(50, color='gray', ls='--', lw=0.8, alpha=0.6, label='50% reference')
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def run_ttt():
    print("\n--- TTT P1 vs P2 Analysis ---")
    game = TicTacToe()
    ql  = load_ql(os.path.join(config.MODEL_DIR, 'ttt_ql.pkl'))
    dqn = load_dqn(os.path.join(config.MODEL_DIR, 'ttt_dqn.pt'),
                   config.DQN_INPUT_TTT, config.DQN_OUTPUT_TTT, config.DQN_HIDDEN_TTT)
    mm  = MinimaxAgent(player=BaseGame.PLAYER2, use_alpha_beta=True)
    df  = DefaultOpponent(player=BaseGame.PLAYER2)
    rnd = RandomAgent(player=BaseGame.PLAYER2)

    matchups = [
        ('Minimax+AB\nvs Default', mm,  df),
        ('DQN\nvs Default',        dqn, df),
        ('QL\nvs Default',         ql,  df),
        ('DQN\nvs Random',         dqn, rnd),
        ('QL\nvs Random',          ql,  rnd),
        ('DQN\nvs QL',             dqn, ql),
        ('Minimax+AB\nvs QL',      mm,  ql),
        ('Minimax+AB\nvs DQN',     mm,  dqn),
    ]

    labels, p1s, p2s, drs = [], [], [], []
    for label, a1, a2 in matchups:
        p1, p2, dr = first_mover_data(a1, a2, game, n=100)
        labels.append(label); p1s.append(p1); p2s.append(p2); drs.append(dr)
        print(f"  {label.replace(chr(10),' '):30s}  P1-wins={p1:.0f}%  P2-wins={p2:.0f}%  Draws={dr:.0f}%")

    plot_p1p2(labels, p1s, p2s, drs,
              title='TTT: Win Rate as P1 (first mover) vs P2 (second mover)',
              path=os.path.join(RESULTS_TTT, 'p1_vs_p2.png'))


def run_c4():
    print("\n--- C4 P1 vs P2 Analysis ---")
    game = Connect4()
    ql  = load_ql(os.path.join(config.MODEL_DIR, 'c4_ql.pkl'))
    dqn = load_dqn(os.path.join(config.MODEL_DIR, 'c4_dqn.pt'),
                   config.DQN_INPUT_C4, config.DQN_OUTPUT_C4, config.DQN_HIDDEN_C4)
    mm  = MinimaxAgent(player=BaseGame.PLAYER2, use_alpha_beta=True,
                       depth_limit=config.MINIMAX_DEPTH_C4, eval_fn=connect4_eval)
    df  = DefaultOpponent(player=BaseGame.PLAYER2)
    rnd = RandomAgent(player=BaseGame.PLAYER2)

    matchups = [
        ('Minimax+AB\nvs Default',  mm,  df),
        ('Minimax+AB\nvs Random',   mm,  rnd),
        ('DQN\nvs Random',          dqn, rnd),
        ('QL\nvs Random',           ql,  rnd),
        ('DQN\nvs Default',         dqn, df),
        ('QL\nvs Default',          ql,  df),
        ('DQN\nvs QL',              dqn, ql),
        ('Minimax+AB\nvs DQN',      mm,  dqn),
        ('Minimax+AB\nvs QL',       mm,  ql),
    ]

    labels, p1s, p2s, drs = [], [], [], []
    for label, a1, a2 in matchups:
        p1, p2, dr = first_mover_data(a1, a2, game, n=50)
        labels.append(label); p1s.append(p1); p2s.append(p2); drs.append(dr)
        print(f"  {label.replace(chr(10),' '):30s}  P1-wins={p1:.0f}%  P2-wins={p2:.0f}%  Draws={dr:.0f}%")

    plot_p1p2(labels, p1s, p2s, drs,
              title='Connect 4: Win Rate as P1 (first mover) vs P2 (second mover)',
              path=os.path.join(RESULTS_C4, 'p1_vs_p2.png'))


if __name__ == '__main__':
    run_ttt()
    run_c4()
    print("\nDone.")
