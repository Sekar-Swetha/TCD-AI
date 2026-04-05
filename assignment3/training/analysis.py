import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.minimax import MinimaxAgent, connect4_eval, TimeLimitExceeded
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
import config
from visualizer import (
    render_ttt, render_ttt_sequence,
    render_connect4, render_connect4_sequence,
    plot_training_curves_detailed,
    plot_game_length_distribution,
    plot_first_mover_advantage,
    plot_ttt_qvalue_heatmap,
)

RESULTS_TTT = os.path.join('results', 'ttt')
RESULTS_C4  = os.path.join('results', 'connect4')
MODEL_DIR   = config.MODEL_DIR


class RandomAgent:
    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
    def select_action(self, game, legal_moves=None):
        return random.choice(legal_moves or game.get_legal_moves())


def _load_ttt_agents():
    mm_ab = MinimaxAgent(player=BaseGame.PLAYER1, use_alpha_beta=True)
    ql = QLearningAgent(player=BaseGame.PLAYER1, epsilon=0.0, epsilon_min=0.0)
    ql.load(os.path.join(MODEL_DIR, 'ttt_ql.pkl'))
    dqn = DQNAgent(player=BaseGame.PLAYER1, input_size=9, output_size=9,
                   hidden_sizes=config.DQN_HIDDEN_TTT, epsilon=0.0, epsilon_min=0.0)
    dqn.load(os.path.join(MODEL_DIR, 'ttt_dqn.pt'))
    return mm_ab, ql, dqn


def _load_c4_agents():
    mm_ab = MinimaxAgent(player=BaseGame.PLAYER1, use_alpha_beta=True,
                         depth_limit=config.MINIMAX_DEPTH_C4, eval_fn=connect4_eval)
    ql = QLearningAgent(player=BaseGame.PLAYER1, epsilon=0.0, epsilon_min=0.0)
    ql.load(os.path.join(MODEL_DIR, 'c4_ql.pkl'))
    dqn = DQNAgent(player=BaseGame.PLAYER1, input_size=42, output_size=7,
                   hidden_sizes=config.DQN_HIDDEN_C4, epsilon=0.0, epsilon_min=0.0)
    dqn.load(os.path.join(MODEL_DIR, 'c4_dqn.pt'))
    return mm_ab, ql, dqn


def _play_and_record(game, agent1, agent2, n=300):
    lengths, p1_wins_first, p1_wins_second, p2_wins_first, p2_wins_second, draws = \
        [], 0, 0, 0, 0, 0

    for i in range(n):
        game.reset()
        first = BaseGame.PLAYER1 if i % 2 == 0 else BaseGame.PLAYER2
        a1_role = first
        a2_role = -first

        agent1.player = a1_role
        agent2.player = a2_role

        moves = 0
        while not game.is_terminal():
            cur = game.get_current_player()
            move = agent1.select_action(game) if cur == a1_role else agent2.select_action(game)
            game.make_move(move)
            moves += 1

        lengths.append(moves)
        winner = game.check_winner()
        if winner == a1_role:
            if a1_role == BaseGame.PLAYER1: p1_wins_first += 1
            else: p1_wins_second += 1
        elif winner == a2_role:
            if a2_role == BaseGame.PLAYER1: p2_wins_first += 1
            else: p2_wins_second += 1
        else:
            draws += 1

    return lengths, p1_wins_first, p1_wins_second, p2_wins_first, p2_wins_second, draws


def analyze_game_lengths_ttt():
    mm_ab, ql, dqn = _load_ttt_agents()
    opp = DefaultOpponent(player=BaseGame.PLAYER2)
    game = TicTacToe()

    matchups = {
        'Minimax+AB vs Default': (mm_ab, opp),
        'Q-Learning vs Default': (ql, opp),
        'DQN vs Default':        (dqn, opp),
        'Minimax+AB vs DQN':     (mm_ab, dqn),
        'Q-Learning vs DQN':     (ql, dqn),
    }

    length_data = {}
    for label, (a1, a2) in matchups.items():
        lengths, *_ = _play_and_record(game, a1, a2, n=300)
        length_data[label] = lengths
        print(f"  {label}: mean={np.mean(lengths):.1f}, min={min(lengths)}, max={max(lengths)}")

    plot_game_length_distribution(
        length_data,
        title='TTT: Game Length Distribution by Matchup',
        path=os.path.join(RESULTS_TTT, 'game_lengths.png'),
    )


def analyze_first_mover_ttt():
    mm_ab, ql, dqn = _load_ttt_agents()
    opp = DefaultOpponent(player=BaseGame.PLAYER2)
    game = TicTacToe()
    N = 400

    matchups = [
        ('MM+AB vs Default', mm_ab, opp),
        ('QL vs Default',    ql,    opp),
        ('DQN vs Default',   dqn,   opp),
        ('MM+AB vs QL',      mm_ab, ql),
        ('MM+AB vs DQN',     mm_ab, dqn),
        ('QL vs DQN',        ql,    dqn),
    ]

    labels, p1_wins_l, p2_wins_l, draws_l = [], [], [], []

    for label, a1, a2 in matchups:
        _, w1f, w1s, w2f, w2s, d = _play_and_record(game, a1, a2, n=N)
        p1_wins_pct = (w1f + w2s) / N * 100
        p2_wins_pct = (w1s + w2f) / N * 100
        draws_pct   = d / N * 100
        labels.append(label)
        p1_wins_l.append(p1_wins_pct)
        p2_wins_l.append(p2_wins_pct)
        draws_l.append(draws_pct)
        print(f"{label}: P1-wins={p1_wins_pct:.1f}%  P2-wins={p2_wins_pct:.1f}%  draws={draws_pct:.1f}%")

    plot_first_mover_advantage(
        labels, p1_wins_l, p2_wins_l, draws_l,
        title='TTT: First-Mover (P1) vs Second-Mover (P2) Win Rate',
        path=os.path.join(RESULTS_TTT, 'first_mover_advantage.png'),
    )


def analyze_qvalues_ttt():
    ql = QLearningAgent(player=BaseGame.PLAYER1, epsilon=0.0, epsilon_min=0.0)
    ql.load(os.path.join(MODEL_DIR, 'ttt_ql.pkl'))

    plot_ttt_qvalue_heatmap(
        ql.q_table, BaseGame.PLAYER1,
        title='Q-Learning: Q-Values on Empty TTT Board (P1 perspective)',
        path=os.path.join(RESULTS_TTT, 'ql_qvalue_heatmap.png'),
    )


def save_game_sequences_ttt():
    mm_ab, ql, dqn = _load_ttt_agents()
    opp = DefaultOpponent(player=BaseGame.PLAYER2)
    game = TicTacToe()

    for (a1, a2, n1, n2) in [
        (mm_ab, opp, 'Minimax+AB', 'Default'),
        (ql, dqn, 'Q-Learning', 'DQN'),
    ]:
        a1.player = BaseGame.PLAYER1
        a2.player = BaseGame.PLAYER2
        game.reset()

        states, moves_taken = [game.get_state()], [None]
        while not game.is_terminal():
            cur = game.get_current_player()
            move = a1.select_action(game) if cur == BaseGame.PLAYER1 else a2.select_action(game)
            game.make_move(move)
            states.append(game.get_state())
            moves_taken.append(move)

        step = max(1, len(states) // 5)
        sel_states = states[::step][:5]
        sel_moves  = moves_taken[::step][:5]

        render_ttt_sequence(
            sel_states, sel_moves,
            title=f'TTT game: {n1} (X) vs {n2} (O)',
            p1_label=f'{n1} (X)', p2_label=f'{n2} (O)',
            path=os.path.join(RESULTS_TTT, f'sequence_{n1}_vs_{n2}.png'.replace('+', '').replace(' ', '_').lower()),
        )


def save_game_sequences_c4():
    mm_ab, ql, dqn = _load_c4_agents()
    rand = RandomAgent(player=BaseGame.PLAYER2)
    game = Connect4()

    for (a1, a2, n1, n2) in [
        (mm_ab, rand, 'Minimax+AB', 'Random'),
        (dqn, rand,   'DQN',        'Random'),
    ]:
        a1.player = BaseGame.PLAYER1
        a2.player = BaseGame.PLAYER2
        game.reset()

        states, moves_taken = [], []
        while not game.is_terminal():
            cur = game.get_current_player()
            move = a1.select_action(game) if cur == BaseGame.PLAYER1 else a2.select_action(game)
            game.make_move(move)
            states.append(game.get_state())
            moves_taken.append(move)

        idxs = np.linspace(0, len(states)-1, 4, dtype=int)
        sel_states = [states[i] for i in idxs]
        sel_moves  = [moves_taken[i] for i in idxs]

        render_connect4_sequence(
            sel_states, sel_moves,
            title=f'C4 game: {n1} (red) vs {n2} (blue)',
            p1_label=f'{n1}', p2_label=f'{n2}',
            path=os.path.join(RESULTS_C4, f'sequence_{n1}_vs_{n2}.png'.replace('+', '').replace(' ', '_').lower()),
        )


def detailed_training_curves_ttt():
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)

    ql = QLearningAgent(player=BaseGame.PLAYER1,
                        alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
                        epsilon=1.0, epsilon_min=0.05, epsilon_decay=config.QL_EPSILON_DECAY)

    ql_res = ql.train(game, opp, num_episodes=10_000, verbose=False)
    plot_training_curves_detailed(
        'Q-Learning vs Default (TTT)', ql_res,
        path=os.path.join(RESULTS_TTT, 'ql_wdl_curve.png'), window=200,
    )

    dqn = DQNAgent(player=BaseGame.PLAYER1, input_size=9, output_size=9,
                   hidden_sizes=config.DQN_HIDDEN_TTT,
                   lr=config.DQN_LR, gamma=config.DQN_GAMMA,
                   epsilon=1.0, epsilon_min=0.05, epsilon_decay=config.DQN_EPSILON_DECAY,
                   batch_size=config.DQN_BATCH_SIZE, buffer_capacity=20000,
                   target_update_freq=500)
    dqn_res = dqn.train(game, opp, num_episodes=10_000, verbose=False)
    plot_training_curves_detailed(
        'DQN vs Default (TTT)', dqn_res,
        path=os.path.join(RESULTS_TTT, 'dqn_wdl_curve.png'), window=200,
    )


def c4_timed_scalability(time_limit_s=120):

    # Theoretical full game tree size estimates
    # Branching factor ≈ 7, avg depth ≈ 36 (games don't always reach 42)
    # Lower bound: 7^36 ≈ 1.5e30 (plain), 7^18 ≈ 1.6e15 (perfect AB)
    THEORETICAL_PLAIN = 7 ** 36
    THEORETICAL_AB    = 7 ** 18

    game = Connect4()
    results = {}

    for label, use_ab in [('Plain Minimax', False), ('Alpha-Beta', True)]:
        game.reset()
        mm = MinimaxAgent(player=BaseGame.PLAYER1, use_alpha_beta=use_ab, depth_limit=None)
        t0 = time.time()
        _, nodes = mm.timed_search(game, time_limit_s)
        elapsed = time.time() - t0
        theoretical = THEORETICAL_AB if use_ab else THEORETICAL_PLAIN
        fraction = nodes / theoretical
        rate = nodes / elapsed
        nodes_30min = rate * 1800
        frac_30min  = nodes_30min / theoretical

        print(f"  {label}:")
        print(f"    Nodes visited in {elapsed:.1f}s: {nodes:,}")
        print(f"    Rate: {rate:,.0f} nodes/s")
        print(f"    Estimated nodes in 30 min: {nodes_30min:,.0f}")
        print(f"    Theoretical full tree size: {theoretical:.2e}")
        print(f"    Fraction covered in 30 min: {frac_30min:.2e} ({frac_30min*100:.10f}%)")
        results[label] = {
            'nodes': nodes, 'elapsed': elapsed, 'rate': rate,
            'nodes_30min': nodes_30min, 'theoretical': theoretical,
            'frac_30min': frac_30min,
        }

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    nodes_30 = [results[l]['nodes_30min'] for l in labels]
    theoretical = [results[l]['theoretical'] for l in labels]

    # log10 to avoid int overflow with huge theoretical values
    log_nodes_30    = [np.log10(max(n, 1)) for n in nodes_30]
    log_theoretical = [np.log10(float(t)) for t in [THEORETICAL_PLAIN, THEORETICAL_AB]]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, log_nodes_30,    0.35, label='log₁₀(nodes in 30 min, extrapolated)', color='#457b9d')
    ax.bar(x + 0.2, log_theoretical, 0.35, label='log₁₀(theoretical full tree)', color='#e63946')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('log₁₀(node count)')
    ax.set_title('C4 Full Minimax: log₁₀ Nodes in 30 min vs Full Tree\n'
                 'Plain≈7^36 nodes, α-β≈7^18 nodes (theoretical)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    for i, l in enumerate(labels):
        frac = results[l]['frac_30min']
        ax.text(i - 0.2, log_nodes_30[i] + 0.3,
                f'{results[l]["nodes_30min"]:,.0f}\nnodes', ha='center', fontsize=7, color='white')
        ax.text(i + 0.2, log_theoretical[i] + 0.3,
                f'10^{log_theoretical[i]:.0f}', ha='center', fontsize=8, color='white')

    fig.tight_layout()
    path = os.path.join(RESULTS_C4, 'c4_timed_scalability.png')
    os.makedirs(RESULTS_C4, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return results


def analyze_game_lengths_c4():
    mm_ab, ql, dqn = _load_c4_agents()
    rand = RandomAgent(player=BaseGame.PLAYER2)
    game = Connect4()

    matchups = {
        'Minimax+AB vs Random': (mm_ab, rand),
        'Q-Learning vs Random': (ql, rand),
        'DQN vs Random':        (dqn, rand),
        'Minimax+AB vs DQN':    (mm_ab, dqn),
    }

    length_data = {}
    for label, (a1, a2) in matchups.items():
        lengths, *_ = _play_and_record(game, a1, a2, n=100)
        length_data[label] = lengths
        print(f"  {label}: mean={np.mean(lengths):.1f}, min={min(lengths)}, max={max(lengths)}")

    plot_game_length_distribution(
        length_data,
        title='Connect 4: Game Length Distribution by Matchup',
        path=os.path.join(RESULTS_C4, 'game_lengths.png'),
    )


def main(ttt=True, c4=True):
    if ttt:
        analyze_game_lengths_ttt()
        analyze_first_mover_ttt()
        analyze_qvalues_ttt()
        save_game_sequences_ttt()
        detailed_training_curves_ttt()

    if c4:
        c4_timed_scalability(time_limit_s=120)  
        analyze_game_lengths_c4()
        save_game_sequences_c4()

    print("\nanalysis complete.")


if __name__ == '__main__':
    main()
