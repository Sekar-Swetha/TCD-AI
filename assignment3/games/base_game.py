from abc import ABC, abstractmethod


class BaseGame(ABC):

    PLAYER1 = 1
    PLAYER2 = -1
    EMPTY = 0

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_legal_moves(self):
        pass

    @abstractmethod
    def make_move(self, move):
        pass

    @abstractmethod
    def undo_move(self):
        pass

    @abstractmethod
    def check_winner(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def get_current_player(self):
        pass

    @abstractmethod
    def render(self):
        pass

    def get_reward(self, player):
        winner = self.check_winner()
        if winner == player:
            return 1.0
        elif winner == 'draw':
            return 0.0
        elif winner is not None:
            return -1.0
        return 0.0
