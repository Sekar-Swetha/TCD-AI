import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from games.base_game import BaseGame
from games.opponent import DefaultOpponent
from agents.minimax import MinimaxAgent, connect4_eval
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from training.evaluate import run_tournament
import config

class RandomAgent:
    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
    def select_action(self, game, legal_moves=None):
        return random.choice(legal_moves or game.get_legal_moves())


def load_ql(path, player=BaseGame.PLAYER1):
    agent = QLearningAgent(player=player,
        alpha=config.QL_ALPHA, gamma=config.QL_GAMMA,
        epsilon=config.QL_EPSILON_MIN, epsilon_min=config.QL_EPSILON_MIN,
        epsilon_decay=config.QL_EPSILON_DECAY)
    agent.load(path)
    return agent


def load_dqn(path, in_size, out_size, hidden, player=BaseGame.PLAYER1):
    agent = DQNAgent(player=player,
        input_size=in_size, output_size=out_size, hidden_sizes=hidden,
        lr=config.DQN_LR, gamma=config.DQN_GAMMA,
        epsilon=config.DQN_EPSILON_MIN, epsilon_min=config.DQN_EPSILON_MIN,
        epsilon_decay=config.DQN_EPSILON_DECAY,
        batch_size=config.DQN_BATCH_SIZE)
    agent.load(path)
    return agent


def fmt(r, name1, name2):
    t = r['total']
    w = r['agent1_wins'] / t * 100
    d = r['draws'] / t * 100
    l = r['agent2_wins'] / t * 100
    return f"  {name1:<22} vs {name2:<22}  W={w:5.1f}%  D={d:5.1f}%  L={l:5.1f}%  ({t} games)"


def run_ttt():

    game = TicTacToe()
    ql_cur  = load_ql(os.path.join(config.MODEL_DIR, 'ttt_ql_curriculum.pkl'))
    dqn_cur = load_dqn(os.path.join(config.MODEL_DIR, 'ttt_dqn_curriculum.pt'),
                       config.DQN_INPUT_TTT, config.DQN_OUTPUT_TTT, config.DQN_HIDDEN_TTT)
    ql_std  = load_ql(os.path.join(config.MODEL_DIR, 'ttt_ql.pkl'))
    dqn_std = load_dqn(os.path.join(config.MODEL_DIR, 'ttt_dqn.pt'),
                       config.DQN_INPUT_TTT, config.DQN_OUTPUT_TTT, config.DQN_HIDDEN_TTT)

    rand_opp    = RandomAgent(player=BaseGame.PLAYER2)
    default_opp = DefaultOpponent(player=BaseGame.PLAYER2)
    mm_opp      = MinimaxAgent(player=BaseGame.PLAYER2, use_alpha_beta=True)

    matchups = [
        (ql_cur,  'QL-Curriculum',  rand_opp,    'Random',       500),
        (ql_cur,  'QL-Curriculum',  default_opp, 'Default',      500),
        (ql_cur,  'QL-Curriculum',  mm_opp,      'Minimax+AB',    100),
        (dqn_cur, 'DQN-Curriculum', rand_opp,    'Random',       500),
        (dqn_cur, 'DQN-Curriculum', default_opp, 'Default',      500),
        (dqn_cur, 'DQN-Curriculum', mm_opp,      'Minimax+AB',    100),
        (ql_cur,  'QL-Curriculum',  dqn_cur,     'DQN-Curriculum',200),
        (ql_cur,  'QL-Curriculum',  ql_std,      'QL-Standard',   200),
        (dqn_cur, 'DQN-Curriculum', dqn_std,     'DQN-Standard',  200),
        (ql_std,  'QL-Standard',    default_opp, 'Default',       500),
        (dqn_std, 'DQN-Standard',   default_opp, 'Default',       500),
    ]

    print()
    for a1, n1, a2, n2, n in matchups:
        a2.player = BaseGame.PLAYER2
        r = run_tournament(a1, a2, game, n_games=n, swap_players=True)
        print(fmt(r, n1, n2))


def run_c4():

    game = Connect4()
    ql_cur  = load_ql(os.path.join(config.MODEL_DIR, 'c4_ql_curriculum.pkl'))
    dqn_cur = load_dqn(os.path.join(config.MODEL_DIR, 'c4_dqn_curriculum.pt'),
                       config.DQN_INPUT_C4, config.DQN_OUTPUT_C4, config.DQN_HIDDEN_C4)
    ql_std  = load_ql(os.path.join(config.MODEL_DIR, 'c4_ql.pkl'))
    dqn_std = load_dqn(os.path.join(config.MODEL_DIR, 'c4_dqn.pt'),
                       config.DQN_INPUT_C4, config.DQN_OUTPUT_C4, config.DQN_HIDDEN_C4)

    rand_opp    = RandomAgent(player=BaseGame.PLAYER2)
    default_opp = DefaultOpponent(player=BaseGame.PLAYER2)
    mm_opp      = MinimaxAgent(player=BaseGame.PLAYER2, use_alpha_beta=True,
                               depth_limit=config.MINIMAX_DEPTH_C4, eval_fn=connect4_eval)

    matchups = [
        (ql_cur,  'QL-Curriculum',  rand_opp,    'Random',        200),
        (ql_cur,  'QL-Curriculum',  default_opp, 'Default',       200),
        (ql_cur,  'QL-Curriculum',  mm_opp,      'Minimax+AB(d3)', 100),
        (dqn_cur, 'DQN-Curriculum', rand_opp,    'Random',        200),
        (dqn_cur, 'DQN-Curriculum', default_opp, 'Default',       200),
        (dqn_cur, 'DQN-Curriculum', mm_opp,      'Minimax+AB(d3)', 100),
        (ql_cur,  'QL-Curriculum',  dqn_cur,     'DQN-Curriculum', 100),
        (ql_cur,  'QL-Curriculum',  ql_std,      'QL-Standard',    100),
        (dqn_cur, 'DQN-Curriculum', dqn_std,     'DQN-Standard',   100),
        (ql_std,  'QL-Standard',    default_opp, 'Default',        200),
        (dqn_std, 'DQN-Standard',   default_opp, 'Default',        200),
    ]

    print()
    for a1, n1, a2, n2, n in matchups:
        a2.player = BaseGame.PLAYER2
        r = run_tournament(a1, a2, game, n_games=n, swap_players=True)
        print(fmt(r, n1, n2))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--game', choices=['ttt', 'connect4', 'both'], default='both')
    args = p.parse_args()

    if args.game in ('ttt', 'both'):
        run_ttt()
    if args.game in ('connect4', 'both'):
        run_c4()

    print("\nDone.")
