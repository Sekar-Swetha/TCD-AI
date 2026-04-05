import random
import pickle
import os
from collections import defaultdict
from agents.base_agent import BaseAgent
from games.base_game import BaseGame


class QLearningAgent(BaseAgent):

    def __init__(self, player=BaseGame.PLAYER1,
                 alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9995):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: defaultdict(float))

        self.training = False

    def select_action(self, game, legal_moves=None):
        legal = legal_moves or game.get_legal_moves()
        state = self._norm(game.get_state())

        if self.training and random.random() < self.epsilon:
            return random.choice(legal)

        q_vals = {a: self.q_table[state][a] for a in legal}
        max_q = max(q_vals.values())
        best = [a for a, v in q_vals.items() if v == max_q]
        return random.choice(best)

    def train(self, game, opponent, num_episodes=50000, verbose=True):
        self.training = True
        results = []

        for ep in range(num_episodes):
            game.reset()

            if random.random() < 0.5:
                first_player = self.player
            else:
                first_player = -self.player
                opp_move = opponent.select_action(game)
                game.make_move(opp_move)

            while not game.is_terminal():
                state = self._norm(game.get_state())
                legal = game.get_legal_moves()
                if not legal:
                    break

                action = self.select_action(game, legal)
                _, reward, done = game.make_move(action)

                if done:
                    winner = game.check_winner()
                    if winner == self.player:
                        r = 1.0
                    elif winner == 'draw':
                        r = 0.2
                    else:
                        r = -1.0
                    self._update(state, action, r, None, [])
                    break

                opp_legal = game.get_legal_moves()
                if not opp_legal:
                    break
                opp_move = opponent.select_action(game)
                _, _, done = game.make_move(opp_move)

                next_state = self._norm(game.get_state())
                next_legal = game.get_legal_moves()

                if done:
                    winner = game.check_winner()
                    if winner == self.player:
                        r = 1.0
                    elif winner == 'draw':
                        r = 0.2
                    else:
                        r = -1.0
                    self._update(state, action, r, None, [])
                    break
                else:
                    self._update(state, action, 0.0, next_state, next_legal)

            winner = game.check_winner()
            if winner == self.player:
                results.append('win')
            elif winner == 'draw':
                results.append('draw')
            else:
                results.append('loss')

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if verbose and (ep + 1) % 5000 == 0:
                last = results[-5000:]
                w = last.count('win') / len(last) * 100
                d = last.count('draw') / len(last) * 100
                print(f"  Episode {ep+1:>6} | ε={self.epsilon:.4f} | "
                      f"win={w:.1f}%  draw={d:.1f}%")

        self.training = False
        return results

    def _norm(self, state):
        return tuple(v * self.player for v in state)

    def _update(self, state, action, reward, next_state, next_legal):
        current_q = self.q_table[state][action]
        if next_state is None or not next_legal:
            target = reward
        else:
            max_next = max(self.q_table[next_state][a] for a in next_legal)
            target = reward + self.gamma * max_next
        self.q_table[state][action] += self.alpha * (target - current_q)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
            }, f)
        print(f"Q-table saved to {path}  ({len(self.q_table)} states)")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in data['q_table'].items():
            for action, val in actions.items():
                self.q_table[state][action] = val
        self.epsilon = data.get('epsilon', self.epsilon_min)
        print(f"Q-table loaded from {path}  ({len(self.q_table)} states)")
