import numpy as np
from games.base_game import BaseGame

WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
    [0, 4, 8], [2, 4, 6],             # diagonals
]

class TicTacToe(BaseGame):

    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = BaseGame.PLAYER1
        self._history = []

    def reset(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = BaseGame.PLAYER1
        self._history = []
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def get_legal_moves(self):
        return [i for i in range(9) if self.board[i] == BaseGame.EMPTY]

    def make_move(self, move):
        assert self.board[move] == BaseGame.EMPTY, f"Illegal move: cell {move} is occupied"
        self._history.append((move, self.current_player))
        self.board[move] = self.current_player
        self.current_player = -self.current_player
        done = self.is_terminal()
        reward = self.get_reward(-self.current_player)  # reward for the player who just moved
        return self.get_state(), reward, done

    def undo_move(self):
        if not self._history:
            return
        move, player = self._history.pop()
        self.board[move] = BaseGame.EMPTY
        self.current_player = player

    def check_winner(self):
        for line in WIN_LINES:
            s = self.board[line].sum()
            if s == 3:
                return BaseGame.PLAYER1
            if s == -3:
                return BaseGame.PLAYER2
        if BaseGame.EMPTY not in self.board:
            return 'draw'
        return None

    def is_terminal(self):
        return self.check_winner() is not None

    def get_current_player(self):
        return self.current_player

    def render(self):
        symbols = {BaseGame.PLAYER1: 'X', BaseGame.PLAYER2: 'O', BaseGame.EMPTY: '.'}
        rows = []
        for r in range(3):
            row = ' '.join(symbols[self.board[r * 3 + c]] for c in range(3))
            rows.append(row)
        print('\n'.join(rows))
        print()
