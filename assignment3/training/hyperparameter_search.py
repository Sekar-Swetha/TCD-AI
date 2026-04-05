import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from games.tictactoe import TicTacToe
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from training.evaluate import run_tournament
from visualizer import plot_hyperparam_sensitivity, plot_hyperparam_heatmap

RESULTS_HP = os.path.join('results', 'ttt', 'hyperparams')
os.makedirs(RESULTS_HP, exist_ok=True)

TRAIN_EPS  = 15_000   # episodes per config (fast but enough to distinguish)
EVAL_GAMES = 200      # games per eval


def eval_win_rate(agent, n_games=EVAL_GAMES):
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)
    r = run_tournament(agent, opp, game, n_games=n_games, swap_players=True)
    return r['agent1_wins'] / r['total'] * 100


def ql_alpha_gamma_search():
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5]
    gammas = [0.80, 0.90, 0.95, 0.99]
    win_matrix = np.zeros((len(alphas), len(gammas)))
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)

    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            agent = QLearningAgent(player=BaseGame.PLAYER1,
                                   alpha=alpha, gamma=gamma,
                                   epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995)
            agent.train(game, opp, num_episodes=TRAIN_EPS, verbose=False)
            win_matrix[i, j] = eval_win_rate(agent)
            print(f"  alpha={alpha}  gamma={gamma}  → win={win_matrix[i,j]:.1f}%")

    plot_hyperparam_heatmap(
        'alpha (learning rate)', alphas,
        'gamma (discount)',      gammas,
        win_matrix,
        title='Q-Learning: Win Rate (%) vs α and γ — TTT vs Default (15k train)',
        path=os.path.join(RESULTS_HP, 'ql_alpha_gamma.png'),
    )
    best_i, best_j = np.unravel_index(win_matrix.argmax(), win_matrix.shape)
    print(f"  Best: alpha={alphas[best_i]}, gamma={gammas[best_j]}, win={win_matrix[best_i,best_j]:.1f}%")
    return alphas[best_i], gammas[best_j]


def ql_epsilon_decay_search():
    decays = [0.999, 0.9995, 0.9999, 0.99995]
    win_rates = []
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)

    for decay in decays:
        agent = QLearningAgent(player=BaseGame.PLAYER1,
                               alpha=0.1, gamma=0.95,
                               epsilon=1.0, epsilon_min=0.05, epsilon_decay=decay)
        agent.train(game, opp, num_episodes=TRAIN_EPS, verbose=False)
        wr = eval_win_rate(agent)
        win_rates.append(wr)
        print(f"  epsilon_decay={decay}  → win={wr:.1f}%")

    plot_hyperparam_sensitivity(
        'epsilon_decay', decays,
        {'Q-Learning win rate': win_rates},
        title='Q-Learning: Win Rate vs Epsilon Decay — TTT (15k train)',
        path=os.path.join(RESULTS_HP, 'ql_epsilon_decay.png'),
    )
    best = decays[int(np.argmax(win_rates))]
    print(f"  Best: epsilon_decay={best}, win={max(win_rates):.1f}%")
    return best


def dqn_lr_hidden_search():
    lrs     = [1e-4, 5e-4, 1e-3, 5e-3]
    hiddens = [(64, 64), (128, 64), (128, 128)]
    hidden_labels = ['(64,64)', '(128,64)', '(128,128)']
    win_matrix = np.zeros((len(lrs), len(hiddens)))
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)

    for i, lr in enumerate(lrs):
        for j, hidden in enumerate(hiddens):
            agent = DQNAgent(player=BaseGame.PLAYER1,
                             input_size=9, output_size=9,
                             hidden_sizes=hidden, lr=lr, gamma=0.95,
                             epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                             batch_size=64, buffer_capacity=20000, target_update_freq=500)
            agent.train(game, opp, num_episodes=TRAIN_EPS, verbose=False)
            win_matrix[i, j] = eval_win_rate(agent)
            print(f"  lr={lr}  hidden={hidden_labels[j]}  → win={win_matrix[i,j]:.1f}%")

    plot_hyperparam_heatmap(
        'learning rate', lrs,
        'hidden sizes', hidden_labels,
        win_matrix,
        title='DQN: Win Rate (%) vs LR and Hidden Sizes — TTT vs Default (15k train)',
        path=os.path.join(RESULTS_HP, 'dqn_lr_hidden.png'),
    )
    best_i, best_j = np.unravel_index(win_matrix.argmax(), win_matrix.shape)
    print(f"  Best: lr={lrs[best_i]}, hidden={hidden_labels[best_j]}, win={win_matrix[best_i,best_j]:.1f}%")
    return lrs[best_i], hiddens[best_j]


def dqn_gamma_search():
    gammas = [0.80, 0.90, 0.95, 0.99]
    win_rates = []
    game = TicTacToe()
    opp  = DefaultOpponent(player=BaseGame.PLAYER2)

    for gamma in gammas:
        agent = DQNAgent(player=BaseGame.PLAYER1,
                         input_size=9, output_size=9,
                         hidden_sizes=(128, 64), lr=1e-3, gamma=gamma,
                         epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                         batch_size=64, buffer_capacity=20000, target_update_freq=500)
        agent.train(game, opp, num_episodes=TRAIN_EPS, verbose=False)
        wr = eval_win_rate(agent)
        win_rates.append(wr)
        print(f"  gamma={gamma}  → win={wr:.1f}%")

    plot_hyperparam_sensitivity(
        'gamma (discount factor)', gammas,
        {'DQN win rate': win_rates},
        title='DQN: Win Rate vs Gamma — TTT (15k train)',
        path=os.path.join(RESULTS_HP, 'dqn_gamma.png'),
    )
    best = gammas[int(np.argmax(win_rates))]
    print(f"  Best: gamma={best}, win={max(win_rates):.1f}%")
    return best


def main():
    print("Hyperparameter grid searches on TTT\n")

    ql_alpha_gamma_search()
    ql_epsilon_decay_search()
    dqn_lr_hidden_search()
    dqn_gamma_search()

    print(f"\nAll hyperparameter charts saved to {RESULTS_HP}")


if __name__ == '__main__':
    main()
