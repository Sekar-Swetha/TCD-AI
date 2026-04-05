import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from games.tictactoe import TicTacToe
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.minimax import MinimaxAgent
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from training.evaluate import run_tournament, print_results
from training.results_logger import (
    save_tournament_csv, tournament_to_row,
    save_training_csv, save_minimax_timing_csv,
)
from visualizer import (
    plot_training_curves_combined, plot_training_curves_detailed,
    plot_vs_default, plot_head_to_head_matrix,
    render_ttt, render_ttt_sequence,
    plot_dqn_loss_curve,
)
import config

RESULTS_TTT = os.path.join('results', 'ttt')
os.makedirs(RESULTS_TTT, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

AGENT_NAMES = ['Minimax', 'Minimax+AB', 'Q-Learning', 'DQN']
MM_EVAL_GAMES = 20  

class RandomAgent:
    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
    def select_action(self, game, legal_moves=None):
        return random.choice(legal_moves or game.get_legal_moves())


def make_minimax(player, use_ab):
    return MinimaxAgent(player=player, use_alpha_beta=use_ab, depth_limit=None)

def make_ql(player, path=None):
    agent = QLearningAgent(player=player,
                           alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
                           epsilon=config.QL_EPSILON, epsilon_min=config.QL_EPSILON_MIN,
                           epsilon_decay=config.QL_EPSILON_DECAY)
    if path and os.path.exists(path):
        agent.load(path)
    return agent

def make_dqn(player, path=None):
    agent = DQNAgent(player=player,
                     input_size=config.DQN_INPUT_TTT, output_size=config.DQN_OUTPUT_TTT,
                     hidden_sizes=config.DQN_HIDDEN_TTT,
                     lr=config.DQN_LR, gamma=config.DQN_GAMMA,
                     epsilon=config.DQN_EPSILON, epsilon_min=config.DQN_EPSILON_MIN,
                     epsilon_decay=config.DQN_EPSILON_DECAY,
                     batch_size=config.DQN_BATCH_SIZE, buffer_capacity=config.DQN_BUFFER_CAPACITY,
                     target_update_freq=config.DQN_TARGET_UPDATE)
    if path and os.path.exists(path):
        agent.load(path)
    return agent


def minimax_timing_analysis():
    game = TicTacToe()
    sequences = [[], [4], [4,0], [4,0,2], [4,0,2,6]] 
    labels    = [f'{len(s)} moves played' for s in sequences]

    timing_rows = []
    plain_nodes, ab_nodes = [], []
    plain_times, ab_times = [], []

    for seq, label in zip(sequences, labels):
        game.reset()
        for m in seq:
            game.make_move(m)

        for use_ab, variant in [(False, 'Plain'), (True, 'Alpha-Beta')]:
            mm = MinimaxAgent(player=BaseGame.PLAYER1, use_alpha_beta=use_ab)
            t0 = time.time()
            mm.select_action(game)
            elapsed = time.time() - t0
            timing_rows.append({
                'game': 'TTT', 'variant': variant, 'depth_or_note': label,
                'nodes_visited': mm.nodes_visited, 'time_s': round(elapsed, 4),
                'nodes_per_sec': round(mm.nodes_visited / max(elapsed, 1e-6)),
            })
            if use_ab:
                ab_nodes.append(mm.nodes_visited)
                ab_times.append(elapsed)
            else:
                plain_nodes.append(mm.nodes_visited)
                plain_times.append(elapsed)

        print(f"  {label}: Plain={plain_nodes[-1]:>8,} nodes {plain_times[-1]:.3f}s | "
              f"AB={ab_nodes[-1]:>6,} nodes {ab_times[-1]:.4f}s | "
              f"speedup={plain_times[-1]/max(ab_times[-1],1e-9):.0f}x")

    save_minimax_timing_csv(timing_rows, os.path.join(RESULTS_TTT, 'minimax_timing.csv'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = range(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.bar([i-w/2 for i in x], plain_nodes, w, label='Plain',     color='#e76f51')
    ax1.bar([i+w/2 for i in x], ab_nodes,    w, label='Alpha-Beta', color='#2a9d8f')
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax1.set_ylabel('Nodes visited'); ax1.set_title('Minimax: Nodes Visited per Move')
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)

    ax2.bar([i-w/2 for i in x], plain_times, w, label='Plain',     color='#e76f51')
    ax2.bar([i+w/2 for i in x], ab_times,    w, label='Alpha-Beta', color='#2a9d8f')
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax2.set_ylabel('Time (s)'); ax2.set_title('Minimax: Time per Move')
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('TTT Minimax: Plain vs Alpha-Beta Pruning\n'
                 '(Both produce IDENTICAL moves — only speed differs)', fontsize=10)
    fig.tight_layout()
    path = os.path.join(RESULTS_TTT, 'minimax_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_rl_agents(ql_path, dqn_path, force=False):
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)

    ql_results, dqn_results = None, None

    if force or not os.path.exists(ql_path):
        ql = make_ql(BaseGame.PLAYER1)
        ql_results = ql.train(game, opp, num_episodes=config.QL_EPISODES_TTT, verbose=True)
        ql.save(ql_path)
        save_training_csv(config.QL_EPISODES_TTT, ql_results, 'Q-Learning', 'TTT',
                          os.path.join(RESULTS_TTT, 'ql_training.csv'))
        plot_training_curves_detailed('Q-Learning vs Default (TTT)', ql_results,
                                      path=os.path.join(RESULTS_TTT, 'ql_wdl_curve.png'))
    else:
        print("already trained, skip")

    if force or not os.path.exists(dqn_path):
        dqn = make_dqn(BaseGame.PLAYER1)
        dqn_results = dqn.train(game, opp, num_episodes=config.DQN_EPISODES_TTT, verbose=True)
        dqn.save(dqn_path)
        save_training_csv(config.DQN_EPISODES_TTT, dqn_results, 'DQN', 'TTT',
                          os.path.join(RESULTS_TTT, 'dqn_training.csv'))
        plot_training_curves_detailed('DQN vs Default (TTT)', dqn_results,
                                      path=os.path.join(RESULTS_TTT, 'dqn_wdl_curve.png'))
        plot_dqn_loss_curve(dqn.loss_history, 'DQN TTT',
                            path=os.path.join(RESULTS_TTT, 'dqn_loss_curve.png'))
    else:
        print("already trained, skip")

    curves = []
    if ql_results:  curves.append(('Q-Learning', ql_results))
    if dqn_results: curves.append(('DQN', dqn_results))
    if curves:
        plot_training_curves_combined(curves, path=os.path.join(RESULTS_TTT, 'training_curves.png'))

    return ql_results, dqn_results


def eval_vs_default(agents, names, n_games_map):
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)
    win_rates, draw_rates, loss_rates = [], [], []
    csv_rows = []

    for agent, name in zip(agents, names):
        n = n_games_map.get(name, config.EVAL_GAMES)
        r = run_tournament(agent, opp, game, n_games=n, swap_players=True)
        print_results(name, 'Default', r)
        t = r['total']
        win_rates.append(r['agent1_wins'] / t * 100)
        draw_rates.append(r['draws'] / t * 100)
        loss_rates.append(r['agent2_wins'] / t * 100)
        csv_rows.append(tournament_to_row('TTT', name, 'Default', r))

    save_tournament_csv(csv_rows, os.path.join(RESULTS_TTT, 'vs_default.csv'))
    return win_rates, draw_rates, loss_rates


def eval_head_to_head(agents, names, n_games_map):
    game = TicTacToe()
    n = len(agents)
    win_matrix = np.full((n, n), 50.0)
    csv_rows = []
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            n_games = min(n_games_map.get(names[i], config.EVAL_GAMES),
                          n_games_map.get(names[j], config.EVAL_GAMES))
            r = run_tournament(agents[i], agents[j], game, n_games=n_games, swap_players=True)
            print_results(names[i], names[j], r)
            win_matrix[i][j] = r['agent1_wins'] / r['total'] * 100
            csv_rows.append(tournament_to_row('TTT', names[i], names[j], r))

    save_tournament_csv(csv_rows, os.path.join(RESULTS_TTT, 'head_to_head.csv'))
    return win_matrix


def save_game_snapshots(agents, names):
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)] 

    for i, j in pairs:
        a1, a2 = agents[i], agents[j]
        n1, n2 = names[i], names[j]
        a1.player = BaseGame.PLAYER1
        a2.player = BaseGame.PLAYER2
        game.reset()

        while not game.is_terminal():
            cur = game.get_current_player()
            move = a1.select_action(game) if cur == BaseGame.PLAYER1 else a2.select_action(game)
            game.make_move(move)

        winner = game.check_winner()
        res = 'Draw' if winner=='draw' else (f'{n1} wins' if winner==BaseGame.PLAYER1 else f'{n2} wins')
        path = os.path.join(RESULTS_TTT,
                            f'game_{n1}_vs_{n2}.png'.replace('+','').replace(' ','_').lower())
        render_ttt(game.get_state(), title=f'{n1}(X) vs {n2}(O) — {res}',
                   p1_label=f'{n1} (X)', p2_label=f'{n2} (O)', path=path)


def main():
    ql_path  = os.path.join(config.MODEL_DIR, 'ttt_ql.pkl')
    dqn_path = os.path.join(config.MODEL_DIR, 'ttt_dqn.pt')

    minimax_timing_analysis()

    train_rl_agents(ql_path, dqn_path)

    mm_plain = make_minimax(BaseGame.PLAYER1, use_ab=False)
    mm_ab    = make_minimax(BaseGame.PLAYER1, use_ab=True)
    ql       = make_ql(BaseGame.PLAYER1, path=ql_path)
    ql.epsilon = 0.0
    dqn      = make_dqn(BaseGame.PLAYER1, path=dqn_path)
    dqn.epsilon = 0.0

    agents = [mm_plain, mm_ab, ql, dqn]
    names  = AGENT_NAMES

    n_games_map = {
        'Minimax':    MM_EVAL_GAMES,
        'Minimax+AB': config.EVAL_GAMES,
        'Q-Learning': config.EVAL_GAMES,
        'DQN':        config.EVAL_GAMES,
    }

    win_rates, draw_rates, loss_rates = eval_vs_default(agents, names, n_games_map)
    note = f'(Minimax: {MM_EVAL_GAMES} games — identical result to Minimax+AB but ~30x slower)'
    plot_vs_default(names, win_rates, draw_rates, loss_rates,
                    title=f'TTT: All 4 Agents vs Default Opponent\n{note}',
                    path=os.path.join(RESULTS_TTT, 'vs_default.png'))

    win_matrix = eval_head_to_head(agents, names, n_games_map)
    plot_head_to_head_matrix(names, win_matrix,
                             title='TTT: Head-to-Head Win Rates (%) — All 4 Agents',
                             path=os.path.join(RESULTS_TTT, 'head_to_head.png'))

    save_game_snapshots(agents, names)

    print(f"\nAll TTT results saved to {RESULTS_TTT}")


if __name__ == '__main__':
    main()
