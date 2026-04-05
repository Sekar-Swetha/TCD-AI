import random
import os
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from games.base_game import BaseGame


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, legal_next):
        self.buf.append((state, action, reward, next_state, done, legal_next))

    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


class DQNAgent(BaseAgent):

    def __init__(self, player=BaseGame.PLAYER1,
                 input_size=9, output_size=9,
                 hidden_sizes=(128, 128),
                 lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 batch_size=64, buffer_capacity=50000,
                 target_update_freq=500):

        self.player = player
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.output_size = output_size
        self.training = False
        self._step = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = QNetwork(input_size, output_size, hidden_sizes).to(self.device)
        self.target_net = QNetwork(input_size, output_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity)
        self.loss_history = []

    def select_action(self, game, legal_moves=None):
        legal = legal_moves or game.get_legal_moves()

        if self.training and random.random() < self.epsilon:
            return random.choice(legal)

        state_t = self._state_tensor(game.get_state())
        with torch.no_grad():
            q_vals = self.policy_net(state_t).squeeze(0)

        # Mask illegal moves with -inf
        mask = torch.full((self.output_size,), float('-inf'), device=self.device)
        for a in legal:
            mask[a] = q_vals[a]

        return int(mask.argmax().item())

    def train(self, game, opponent, num_episodes=50000, verbose=True):
        self.training = True
        results = []

        for ep in range(num_episodes):
            game.reset()

            if random.random() < 0.5:
                pass
            else:
                opp_move = opponent.select_action(game)
                game.make_move(opp_move)

            while not game.is_terminal():
                state = game.get_state()
                legal = game.get_legal_moves()
                if not legal:
                    break

                action = self.select_action(game, legal)
                _, _, done = game.make_move(action)

                if done:
                    winner = game.check_winner()
                    r = 1.0 if winner == self.player else (0.2 if winner == 'draw' else -1.0)
                    self.replay.push(state, action, r, None, True, [])
                    self._learn()
                    break

                opp_legal = game.get_legal_moves()
                if not opp_legal:
                    break
                opp_move = opponent.select_action(game)
                _, _, done = game.make_move(opp_move)

                next_state = game.get_state()
                next_legal = game.get_legal_moves()

                if done:
                    winner = game.check_winner()
                    r = 1.0 if winner == self.player else (0.2 if winner == 'draw' else -1.0)
                    self.replay.push(state, action, r, None, True, [])
                else:
                    self.replay.push(state, action, 0.0, next_state, False, next_legal)

                self._learn()

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

    def _learn(self):
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones, legal_nexts = zip(*batch)

        states_t = torch.stack([self._state_tensor(s) for s in states]).squeeze(1).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q = torch.zeros(self.batch_size, device=self.device)
            non_terminal = ~dones_t
            if non_terminal.any():
                ns_list = [s for s, d in zip(next_states, dones) if not d]
                nl_list = [l for l, d in zip(legal_nexts, dones) if not d]
                ns_t = torch.stack([self._state_tensor(s) for s in ns_list]).squeeze(1).to(self.device)
                all_q = self.target_net(ns_t)

                for i, (q_row, legal) in enumerate(zip(all_q, nl_list)):
                    if legal:
                        masked = torch.full((self.output_size,), float('-inf'), device=self.device)
                        for a in legal:
                            masked[a] = q_row[a]
                        idx = non_terminal.nonzero(as_tuple=True)[0][i]
                        next_q[idx] = masked.max()

            targets = rewards_t + self.gamma * next_q * (~dones_t).float()

        loss = nn.MSELoss()(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.loss_history.append((self._step, loss.item()))

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step': self._step,
        }, path)
        print(f"DQN saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt.get('epsilon', self.epsilon_min)
        self._step = ckpt.get('step', 0)
        print(f"DQN loaded from {path}")

    def _state_tensor(self, state):
        arr = np.array(state, dtype=np.float32) * self.player  # flip sign for perspective
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
