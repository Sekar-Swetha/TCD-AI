import numpy as np
from games.base_game import BaseGame


ROWS = 6
COLS = 7

class Connect4(BaseGame):

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = BaseGame.PLAYER1
        self._history = []

    def reset(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = BaseGame.PLAYER1
        self._history = []
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def get_legal_moves(self):
        return [c for c in range(COLS) if self.board[0, c] == BaseGame.EMPTY]

    def _drop_row(self, col):
        for r in range(ROWS - 1, -1, -1):
            if self.board[r, col] == BaseGame.EMPTY:
                return r
        return -1

    def make_move(self, col):
        row = self._drop_row(col)
        assert row != -1, f"Illegal move: column {col} is full"
        self._history.append((col, row, self.current_player))
        self.board[row, col] = self.current_player
        self.current_player = -self.current_player
        done = self.is_terminal()
        reward = self.get_reward(-self.current_player)
        return self.get_state(), reward, done

    def undo_move(self):
        if not self._history:
            return
        col, row, player = self._history.pop()
        self.board[row, col] = BaseGame.EMPTY
        self.current_player = player

    def check_winner(self):
        for r in range(ROWS):
            for c in range(COLS):
                p = self.board[r, c]
                if p == BaseGame.EMPTY:
                    continue
                if c + 3 < COLS and all(self.board[r, c + i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and all(self.board[r + i, c] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c + 3 < COLS and all(self.board[r + i, c + i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c - 3 >= 0 and all(self.board[r + i, c - i] == p for i in range(4)):
                    return p

        if len(self.get_legal_moves()) == 0:
            return 'draw'
        return None

    def is_terminal(self):
        return self.check_winner() is not None

    def get_current_player(self):
        return self.current_player

    def render(self):
        symbols = {BaseGame.PLAYER1: 'X', BaseGame.PLAYER2: 'O', BaseGame.EMPTY: '.'}
        print(' '.join(str(c) for c in range(COLS)))
        for r in range(ROWS):
            print(' '.join(symbols[self.board[r, c]] for c in range(COLS)))
        print()
