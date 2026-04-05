import argparse
import os
import random
import sys

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.minimax import MinimaxAgent, connect4_eval
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from training.evaluate import run_tournament, print_results
import config

MODEL_DIR = config.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)


class RandomAgent:
    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
    def select_action(self, game, legal_moves=None):
        return random.choice(legal_moves or game.get_legal_moves())


class HumanAgent:
    def __init__(self, player=BaseGame.PLAYER1):
        self.player = player
    def select_action(self, game, legal_moves=None):
        legal = legal_moves or game.get_legal_moves()
        game.render()
        while True:
            try:
                move = int(input(f"Your move {legal}: "))
                if move in legal:
                    return move
                print("Illegal move. Try again.")
            except (ValueError, EOFError):
                print("Enter a number.")


def make_agent(name, game_name, player):
    is_c4 = game_name == 'connect4'

    if name == 'minimax':
        depth = None if not is_c4 else config.MINIMAX_DEPTH_C4
        eval_fn = connect4_eval if is_c4 else None
        return MinimaxAgent(player=player, use_alpha_beta=False,
                            depth_limit=depth, eval_fn=eval_fn)

    if name == 'minimax_ab':
        depth = None if not is_c4 else config.MINIMAX_DEPTH_C4
        eval_fn = connect4_eval if is_c4 else None
        return MinimaxAgent(player=player, use_alpha_beta=True,
                            depth_limit=depth, eval_fn=eval_fn)

    if name == 'ql':
        path = os.path.join(MODEL_DIR, f'{"c4" if is_c4 else "ttt"}_ql.pkl')
        agent = QLearningAgent(
            player=player,
            alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
            epsilon=config.QL_EPSILON_MIN,
            epsilon_min=config.QL_EPSILON_MIN,
            epsilon_decay=config.QL_EPSILON_DECAY,
        )
        if os.path.exists(path):
            agent.load(path)
        else:
            print(f"[warn] No trained Q-table at {path}. Run --train ql first.")
        return agent

    if name == 'dqn':
        path = os.path.join(MODEL_DIR, f'{"c4" if is_c4 else "ttt"}_dqn.pt')
        in_size  = config.DQN_INPUT_C4   if is_c4 else config.DQN_INPUT_TTT
        out_size = config.DQN_OUTPUT_C4  if is_c4 else config.DQN_OUTPUT_TTT
        hidden   = config.DQN_HIDDEN_C4  if is_c4 else config.DQN_HIDDEN_TTT
        agent = DQNAgent(
            player=player,
            input_size=in_size, output_size=out_size, hidden_sizes=hidden,
            lr=config.DQN_LR, gamma=config.DQN_GAMMA,
            epsilon=config.DQN_EPSILON_MIN,
            epsilon_min=config.DQN_EPSILON_MIN,
            epsilon_decay=config.DQN_EPSILON_DECAY,
            batch_size=config.DQN_BATCH_SIZE,
        )
        if os.path.exists(path):
            agent.load(path)
        else:
            print(f"[warn] No trained DQN at {path}. Run --train dqn first.")
        return agent

    if name == 'default':
        return DefaultOpponent(player=player)

    if name == 'random':
        return RandomAgent(player=player)

    if name == 'human':
        return HumanAgent(player=player)

    raise ValueError(f"Unknown agent '{name}'. Choose: minimax, minimax_ab, ql, dqn, default, random, human")


def make_game(name):
    if name == 'ttt':
        return TicTacToe()
    if name == 'connect4':
        return Connect4()
    raise ValueError(f"Unknown game '{name}'. Choose: ttt, connect4")


def play(args):
    game   = make_game(args.game)
    agent1 = make_agent(args.agent,    args.game, BaseGame.PLAYER1)
    agent2 = make_agent(args.opponent, args.game, BaseGame.PLAYER2)

    n = args.games
    is_interactive = args.agent == 'human' or args.opponent == 'human'

    if is_interactive:
        n = 1

    results = run_tournament(agent1, agent2, game, n_games=n, swap_players=not is_interactive)

    a1_name = args.agent
    a2_name = args.opponent
    print(f"\n{'='*60}")
    print(f"Results over {n} game(s): {a1_name} (P1) vs {a2_name} (P2)")
    print_results(a1_name, a2_name, results)

    if is_interactive:
        game.render()
        winner = game.check_winner()
        if winner == BaseGame.PLAYER1:
            print(f"Winner: {a1_name} (X)")
        elif winner == BaseGame.PLAYER2:
            print(f"Winner: {a2_name} (O)")
        else:
            print("Draw")


def train(args):
    game_name = args.game
    game = make_game(game_name)
    is_c4 = game_name == 'connect4'

    if is_c4:
        opp = RandomAgent(player=BaseGame.PLAYER2)
        opp_name = 'random'
    else:
        opp = DefaultOpponent(player=BaseGame.PLAYER2)
        opp_name = 'default'

    agent_name = args.train

    if agent_name == 'ql':
        episodes = config.QL_EPISODES_C4 if is_c4 else config.QL_EPISODES_TTT
        path = os.path.join(MODEL_DIR, f'{"c4" if is_c4 else "ttt"}_ql.pkl')
        agent = QLearningAgent(
            player=BaseGame.PLAYER1,
            alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
            epsilon=config.QL_EPSILON, epsilon_min=config.QL_EPSILON_MIN,
            epsilon_decay=config.QL_EPSILON_DECAY,
        )
        print(f"Training Q-Learning on {game_name} for {episodes} episodes vs {opp_name}...")
        agent.train(game, opp, num_episodes=episodes, verbose=True)
        agent.save(path)

    elif agent_name == 'dqn':
        episodes = config.DQN_EPISODES_C4 if is_c4 else config.DQN_EPISODES_TTT
        path = os.path.join(MODEL_DIR, f'{"c4" if is_c4 else "ttt"}_dqn.pt')
        in_size  = config.DQN_INPUT_C4   if is_c4 else config.DQN_INPUT_TTT
        out_size = config.DQN_OUTPUT_C4  if is_c4 else config.DQN_OUTPUT_TTT
        hidden   = config.DQN_HIDDEN_C4  if is_c4 else config.DQN_HIDDEN_TTT
        agent = DQNAgent(
            player=BaseGame.PLAYER1,
            input_size=in_size, output_size=out_size, hidden_sizes=hidden,
            lr=config.DQN_LR, gamma=config.DQN_GAMMA,
            epsilon=config.DQN_EPSILON, epsilon_min=config.DQN_EPSILON_MIN,
            epsilon_decay=config.DQN_EPSILON_DECAY,
            batch_size=config.DQN_BATCH_SIZE, buffer_capacity=config.DQN_BUFFER_CAPACITY,
            target_update_freq=config.DQN_TARGET_UPDATE,
        )
        print(f"Training DQN on {game_name} for {episodes} episodes vs {opp_name}...")
        agent.train(game, opp, num_episodes=episodes, verbose=True)
        agent.save(path)

    else:
        print(f"Unknown agent for training: '{agent_name}'. Choose: ql, dqn")


def eval_all(args):
    if args.game == 'ttt':
        print("Running full TTT evaluation")
        from training.train_ttt import main as ttt_main
        ttt_main()
    elif args.game == 'connect4':
        print("Running full Connect 4 evaluation")
        from training.train_connect4 import main as c4_main
        c4_main()


def main():
    parser = argparse.ArgumentParser(
        description='CS7IS2 Individual Assignment 3 — Game based RL - AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--game',     required=True, choices=['ttt', 'connect4'],
                        help='Game to play')
    parser.add_argument('--agent',    default=None,
                        help='Agent for player 1: minimax, minimax_ab, ql, dqn, human, random, default')
    parser.add_argument('--opponent', default=None,
                        help='Agent for player 2: minimax, minimax_ab, ql, dqn, human, random, default')
    parser.add_argument('--games',    type=int, default=100,
                        help='Number of games to play (default: 100)')
    parser.add_argument('--train',    default=None, choices=['ql', 'dqn'],
                        help='Train an RL agent: ql or dqn')
    parser.add_argument('--eval',     default=None, choices=['all'],
                        help='Run full evaluation suite: all')

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.eval:
        eval_all(args)
    elif args.agent and args.opponent:
        play(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
