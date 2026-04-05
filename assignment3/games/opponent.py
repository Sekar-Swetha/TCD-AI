import random
from games.base_game import BaseGame


class DefaultOpponent:

    def __init__(self, player=BaseGame.PLAYER2):
        self.player = player
        self.opponent = -player

    def _is_winning_move(self, game, move):
        game.make_move(move)
        winner = game.check_winner()
        game.undo_move()
        return winner == self.player

    def _is_blocking_move(self, game, move):
        original_player = game.current_player
        game.current_player = self.opponent
        game.make_move(move)
        winner = game.check_winner()
        game.undo_move()
        game.current_player = original_player
        return winner == self.opponent

    def select_action(self, game):
        legal = game.get_legal_moves()
        if not legal:
            return None

        for move in legal:
            if self._is_winning_move(game, move):
                return move

        for move in legal:
            if self._is_blocking_move(game, move):
                return move
            
        preferred = self._preferred_moves(game, legal)
        return preferred[0] if preferred else random.choice(legal)

    def _preferred_moves(self, game, legal):
        from games.tictactoe import TicTacToe
        from games.connect4 import Connect4

        if isinstance(game, TicTacToe):
            priority = [4, 0, 2, 6, 8, 1, 3, 5, 7]
            return [m for m in priority if m in legal]

        if isinstance(game, Connect4):
            priority = [3, 2, 4, 1, 5, 0, 6]
            return [m for m in priority if m in legal]

        return list(legal)
