import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from games.connect4 import Connect4
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.minimax import MinimaxAgent, connect4_eval, TimeLimitExceeded
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
    render_connect4, render_connect4_sequence,
    plot_dqn_loss_curve,
)
import config

RESULTS_C4 = os.path.join('results', 'connect4')
os.makedirs(RESULTS_C4, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

AGENT_NAMES = ['Minimax(d3)', 'Minimax+AB(d3)', 'Q-Learning', 'DQN']


class RandomAgent:
    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
    def select_action(self, game, legal_moves=None):
        return random.choice(legal_moves or game.get_legal_moves())


def make_minimax_c4(player, use_ab):
    return MinimaxAgent(player=player, use_alpha_beta=use_ab,
                        depth_limit=config.MINIMAX_DEPTH_C4, eval_fn=connect4_eval)

def make_ql_c4(player, path=None):
    agent = QLearningAgent(player=player,
                           alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
                           epsilon=config.QL_EPSILON, epsilon_min=config.QL_EPSILON_MIN,
                           epsilon_decay=config.QL_EPSILON_DECAY)
    if path and os.path.exists(path):
        agent.load(path)
    return agent

def make_dqn_c4(player, path=None):
    agent = DQNAgent(player=player,
                     input_size=config.DQN_INPUT_C4, output_size=config.DQN_OUTPUT_C4,
                     hidden_sizes=config.DQN_HIDDEN_C4,
                     lr=config.DQN_LR, gamma=config.DQN_GAMMA,
                     epsilon=config.DQN_EPSILON, epsilon_min=config.DQN_EPSILON_MIN,
                     epsilon_decay=config.DQN_EPSILON_DECAY,
                     batch_size=config.DQN_BATCH_SIZE, buffer_capacity=config.DQN_BUFFER_CAPACITY,
                     target_update_freq=config.DQN_TARGET_UPDATE)
    if path and os.path.exists(path):
        agent.load(path)
    return agent


def c4_scalability_analysis(timed_seconds=120):
    timing_rows = []

    THEORETICAL_PLAIN = 7 ** 36   # ~2.6e30 (plain)
    THEORETICAL_AB    = 7 ** 18   # ~1.6e15 (alpha-beta)
    game = Connect4()

    timed_results = {}
    for label, use_ab in [('Plain Minimax', False), ('Alpha-Beta', True)]:
        game.reset()
        mm = MinimaxAgent(player=BaseGame.PLAYER1, use_alpha_beta=use_ab, depth_limit=None)
        t0 = time.time()
        _, nodes = mm.timed_search(game, timed_seconds)
        elapsed = time.time() - t0
        rate = nodes / elapsed
        nodes_30min = rate * 1800   # 30 minutes
        theoretical = THEORETICAL_AB if use_ab else THEORETICAL_PLAIN
        frac = nodes_30min / theoretical

        print(f"\n  {label}:")
        print(f"    Nodes in {elapsed:.0f}s:          {nodes:>12,}")
        print(f"    Rate (nodes/sec):          {rate:>12,.0f}")
        print(f"    Extrapolated to 30 min:    {nodes_30min:>12,.0f}")
        print(f"    Theoretical full tree:     {'~'+str(theoretical)[:8]:>12}")
        print(f"    Fraction covered (30 min): {frac:.2e}  ({frac*100:.12f}%)")
        print(f"    → INFEASIBLE to complete full C4 minimax search.")

        timed_results[label] = {'nodes': nodes, 'rate': rate,
                                 'nodes_30min': nodes_30min, 'frac': frac}
        timing_rows.append({
            'game': 'C4', 'variant': label, 'depth_or_note': 'full (no limit)',
            'nodes_visited': nodes, 'time_s': round(elapsed, 2),
            'nodes_per_sec': round(rate),
        })

    depths = list(range(1, 7))
    plain_times, ab_times = [], []
    plain_nodes, ab_nodes = [], []

    for d in depths:
        game.reset()
        for use_ab, lst_t, lst_n in [(False, plain_times, plain_nodes),
                                      (True,  ab_times,   ab_nodes)]:
            mm = MinimaxAgent(player=BaseGame.PLAYER1, use_alpha_beta=use_ab,
                              depth_limit=d, eval_fn=connect4_eval)
            t0 = time.time()
            mm.select_action(game)
            t  = time.time() - t0
            lst_t.append(t)
            lst_n.append(mm.nodes_visited)
            label2 = 'Plain' if not use_ab else 'Alpha-Beta'
            timing_rows.append({
                'game': 'C4', 'variant': label2, 'depth_or_note': f'depth={d}',
                'nodes_visited': mm.nodes_visited, 'time_s': round(t, 4),
                'nodes_per_sec': round(mm.nodes_visited / max(t, 1e-9)),
            })

        print(f"  Depth {d}: plain={plain_nodes[-1]:>8,} {plain_times[-1]:.3f}s | "
              f"AB={ab_nodes[-1]:>6,} {ab_times[-1]:.4f}s | "
              f"speedup={plain_times[-1]/max(ab_times[-1],1e-9):.1f}x")

    save_minimax_timing_csv(timing_rows, os.path.join(RESULTS_C4, 'minimax_timing.csv'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.semilogy(depths, plain_nodes, 'o-', label='Plain', color='#e76f51')
    ax.semilogy(depths, ab_nodes,    's-', label='α-β',   color='#2a9d8f')
    ax.axvline(config.MINIMAX_DEPTH_C4, color='black', ls='--', lw=1,
               label=f'Chosen depth={config.MINIMAX_DEPTH_C4}')
    ax.set_xlabel('Depth'); ax.set_ylabel('Nodes visited (log)')
    ax.set_title('C4: Nodes vs Depth'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(depths, plain_times, 'o-', label='Plain', color='#e76f51')
    ax.semilogy(depths, ab_times,    's-', label='α-β',   color='#2a9d8f')
    ax.axvline(config.MINIMAX_DEPTH_C4, color='black', ls='--', lw=1,
               label=f'Chosen depth={config.MINIMAX_DEPTH_C4}')
    ax.set_xlabel('Depth'); ax.set_ylabel('Time per move (s, log)')
    ax.set_title('C4: Time per Move vs Depth'); ax.legend(); ax.grid(alpha=0.3)

    pn, pt = timed_results['Plain Minimax']['nodes_30min'], timed_results['Plain Minimax']['frac']
    an, at = timed_results['Alpha-Beta']['nodes_30min'],   timed_results['Alpha-Beta']['frac']
    fig.suptitle(f'C4 Minimax Scalability — Full search infeasible\n'
                 f'30-min timed run: plain explores {pt*100:.1e}% of tree, '
                 f'α-β explores {at*100:.1e}%', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_C4, 'minimax_scalability.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # log10 to avoid int overflow with huge theoretical values
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    log_explored  = [np.log10(max(timed_results[l]['nodes_30min'],1)) for l in timed_results]
    log_theory    = [np.log10(float(THEORETICAL_PLAIN)), np.log10(float(THEORETICAL_AB))]
    ax2.bar(x-0.2, log_explored, 0.35, label='Nodes in 30 min (log₁₀)', color='#457b9d')
    ax2.bar(x+0.2, log_theory,   0.35, label='Full tree size (log₁₀)',   color='#e63946')
    ax2.set_xticks(x); ax2.set_xticklabels(['Plain Minimax', 'Alpha-Beta'])
    ax2.set_ylabel('log₁₀(node count)')
    ax2.set_title('C4: Nodes explored in 30 min vs Full tree size\n(Both are completely infeasible)')
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)
    for i, l in enumerate(timed_results):
        fr = timed_results[l]['frac']
        ax2.text(i-0.2, log_explored[i]+0.3, f'{fr*100:.1e}%\nof tree',
                 ha='center', fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_C4, 'c4_timed_scalability.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)


def train_rl_agents(ql_path, dqn_path, force=False):
    game = Connect4()
    rand = RandomAgent(player=BaseGame.PLAYER2)

    ql_results, dqn_results = None, None

    if force or not os.path.exists(ql_path):
        ql = make_ql_c4(BaseGame.PLAYER1)
        ql_results = ql.train(game, rand, num_episodes=config.QL_EPISODES_C4, verbose=True)
        ql.save(ql_path)
        save_training_csv(config.QL_EPISODES_C4, ql_results, 'Q-Learning', 'C4',
                          os.path.join(RESULTS_C4, 'ql_training.csv'))
        plot_training_curves_detailed('Q-Learning vs Random (C4)', ql_results,
                                      path=os.path.join(RESULTS_C4, 'ql_wdl_curve.png'))
    else:
        print("already trained, skip")

    if force or not os.path.exists(dqn_path):
        print(f"\n--- Training DQN ({config.DQN_EPISODES_C4} episodes vs Random) ---")
        dqn = make_dqn_c4(BaseGame.PLAYER1)
        dqn_results = dqn.train(game, rand, num_episodes=config.DQN_EPISODES_C4, verbose=True)
        dqn.save(dqn_path)
        save_training_csv(config.DQN_EPISODES_C4, dqn_results, 'DQN', 'C4',
                          os.path.join(RESULTS_C4, 'dqn_training.csv'))
        plot_training_curves_detailed('DQN vs Random (C4)', dqn_results,
                                      path=os.path.join(RESULTS_C4, 'dqn_wdl_curve.png'))
        plot_dqn_loss_curve(dqn.loss_history, 'DQN C4',
                            path=os.path.join(RESULTS_C4, 'dqn_loss_curve.png'))
    else:
        print("already trained, skip")

    curves = []
    if ql_results:  curves.append(('Q-Learning', ql_results))
    if dqn_results: curves.append(('DQN', dqn_results))
    if curves:
        plot_training_curves_combined(curves, path=os.path.join(RESULTS_C4, 'training_curves.png'))


def eval_vs_random(agents, names, n_games_map):
    game = Connect4()
    rand = RandomAgent(player=BaseGame.PLAYER2)
    win_rates, draw_rates, loss_rates = [], [], []
    csv_rows = []

    print("\n=== Agents vs Random Opponent ===")
    for agent, name in zip(agents, names):
        n = n_games_map.get(name, config.EVAL_GAMES_C4)
        r = run_tournament(agent, rand, game, n_games=n, swap_players=True)
        print_results(name, 'Random', r)
        t = r['total']
        win_rates.append(r['agent1_wins'] / t * 100)
        draw_rates.append(r['draws'] / t * 100)
        loss_rates.append(r['agent2_wins'] / t * 100)
        csv_rows.append(tournament_to_row('C4', name, 'Random', r))

    save_tournament_csv(csv_rows, os.path.join(RESULTS_C4, 'vs_random.csv'))
    return win_rates, draw_rates, loss_rates


def eval_head_to_head(agents, names, n_games_map):
    game = Connect4()
    n = len(agents)
    win_matrix = np.full((n, n), 50.0)
    csv_rows = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            n_games = min(n_games_map.get(names[i], config.EVAL_GAMES_C4),
                          n_games_map.get(names[j], config.EVAL_GAMES_C4))
            r = run_tournament(agents[i], agents[j], game, n_games=n_games, swap_players=True)
            print_results(names[i], names[j], r)
            win_matrix[i][j] = r['agent1_wins'] / r['total'] * 100
            csv_rows.append(tournament_to_row('C4', names[i], names[j], r))

    save_tournament_csv(csv_rows, os.path.join(RESULTS_C4, 'head_to_head.csv'))
    return win_matrix


def save_game_snapshots(agents, names):
    game = Connect4()
    rand = RandomAgent(player=BaseGame.PLAYER2)
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for i, j in pairs:
        a1, a2 = agents[i], agents[j]
        n1, n2 = names[i], names[j]
        a1.player = BaseGame.PLAYER1; a2.player = BaseGame.PLAYER2
        game.reset()
        while not game.is_terminal():
            cur = game.get_current_player()
            move = a1.select_action(game) if cur == BaseGame.PLAYER1 else a2.select_action(game)
            game.make_move(move)
        winner = game.check_winner()
        res = 'Draw' if winner=='draw' else (f'{n1} wins' if winner==BaseGame.PLAYER1 else f'{n2} wins')
        path = os.path.join(RESULTS_C4,
                            f'game_{n1}_vs_{n2}.png'.replace('+','').replace(' ','_').lower())
        render_connect4(game.get_state(), title=f'{n1}(red) vs {n2}(blue) — {res}',
                        p1_label=f'{n1} (red)', p2_label=f'{n2} (blue)', path=path)


def main(run_scalability=True):
    ql_path  = os.path.join(config.MODEL_DIR, 'c4_ql.pkl')
    dqn_path = os.path.join(config.MODEL_DIR, 'c4_dqn.pt')

    if run_scalability:
        c4_scalability_analysis(timed_seconds=120)

    train_rl_agents(ql_path, dqn_path)

    mm_plain = make_minimax_c4(BaseGame.PLAYER1, use_ab=False)
    mm_ab    = make_minimax_c4(BaseGame.PLAYER1, use_ab=True)
    ql       = make_ql_c4(BaseGame.PLAYER1, path=ql_path); ql.epsilon = 0.0
    dqn      = make_dqn_c4(BaseGame.PLAYER1, path=dqn_path); dqn.epsilon = 0.0

    agents = [mm_plain, mm_ab, ql, dqn]
    names  = AGENT_NAMES

    n_games_map = {
        'Minimax(d3)':    100,
        'Minimax+AB(d3)': config.EVAL_GAMES_C4,
        'Q-Learning':     config.EVAL_GAMES_C4,
        'DQN':            config.EVAL_GAMES_C4,
    }

    win_rates, draw_rates, loss_rates = eval_vs_random(agents, names, n_games_map)
    plot_vs_default(names, win_rates, draw_rates, loss_rates,
                    title='C4: All 4 Agents vs Random Opponent\n'
                          '(Minimax: 100 games, others: 200 games)',
                    path=os.path.join(RESULTS_C4, 'vs_random.png'))

    win_matrix = eval_head_to_head(agents, names, n_games_map)
    plot_head_to_head_matrix(names, win_matrix,
                             title='C4: Head-to-Head Win Rates (%) — All 4 Agents',
                             path=os.path.join(RESULTS_C4, 'head_to_head.png'))

    save_game_snapshots(agents, names)

if __name__ == '__main__':
    main(run_scalability=False)
