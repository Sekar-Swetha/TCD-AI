import math
import time
from agents.base_agent import BaseAgent
from games.base_game import BaseGame


class TimeLimitExceeded(Exception):
    pass


class MinimaxAgent(BaseAgent):

    def __init__(self, player=BaseGame.PLAYER1, use_alpha_beta=True,
                 depth_limit=None, eval_fn=None):
        self.player = player
        self.use_alpha_beta = use_alpha_beta
        self.depth_limit = depth_limit
        self.eval_fn = eval_fn or (lambda game, p: 0.0)
        self.nodes_visited = 0
        self._time_limit = None
        self._start_time = None

    def select_action(self, game, legal_moves=None):
        self.nodes_visited = 0
        legal = legal_moves or game.get_legal_moves()
        best_move = None
        best_score = -math.inf

        alpha, beta = -math.inf, math.inf

        for move in legal:
            game.make_move(move)
            score = self._minimax(
                game,
                depth=1,
                is_maximising=False,
                alpha=alpha,
                beta=beta,
            )
            game.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

            if self.use_alpha_beta:
                alpha = max(alpha, best_score)

        return best_move

    def timed_search(self, game, time_limit_s):
        self._time_limit = time_limit_s
        self._start_time = time.time()
        self.nodes_visited = 0
        try:
            move = self.select_action(game)
        except TimeLimitExceeded:
            move = None
        self._time_limit = None
        return move, self.nodes_visited

    def _minimax(self, game, depth, is_maximising, alpha, beta):
        self.nodes_visited += 1
        if self._time_limit and (time.time() - self._start_time) > self._time_limit:
            raise TimeLimitExceeded()

        winner = game.check_winner()
        if winner == self.player:
            return 1.0
        if winner == -self.player:
            return -1.0
        if winner == 'draw':
            return 0.0

        legal = game.get_legal_moves()
        if not legal:
            return 0.0

        if self.depth_limit is not None and depth >= self.depth_limit:
            return self.eval_fn(game, self.player)

        if is_maximising:
            best = -math.inf
            for move in legal:
                game.make_move(move)
                val = self._minimax(game, depth + 1, False, alpha, beta)
                game.undo_move()
                best = max(best, val)
                if self.use_alpha_beta:
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break  # beta cut-off
            return best
        else:
            best = math.inf
            for move in legal:
                game.make_move(move)
                val = self._minimax(game, depth + 1, True, alpha, beta)
                game.undo_move()
                best = min(best, val)
                if self.use_alpha_beta:
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
            return best


def connect4_eval(game, player):

    from games.connect4 import ROWS, COLS
    board = game.board
    opponent = -player

    def score_window(window):
        p_count = (window == player).sum()
        o_count = (window == opponent).sum()
        e_count = (window == 0).sum()
        if p_count == 4:
            return 100
        if p_count == 3 and e_count == 1:
            return 5
        if p_count == 2 and e_count == 2:
            return 2
        if o_count == 3 and e_count == 1:
            return -4
        if o_count == 2 and e_count == 2:
            return -1
        return 0

    import numpy as np
    score = 0

    centre_col = board[:, COLS // 2]
    score += int((centre_col == player).sum()) * 3

    for r in range(ROWS):
        for c in range(COLS - 3):
            w = board[r, c:c + 4]
            score += score_window(w)

    for c in range(COLS):
        for r in range(ROWS - 3):
            w = board[r:r + 4, c]
            score += score_window(w)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = np.array([board[r + i, c + i] for i in range(4)])
            score += score_window(w)

    for r in range(ROWS - 3):
        for c in range(3, COLS):
            w = np.array([board[r + i, c - i] for i in range(4)])
            score += score_window(w)

    return float(score)
