import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from games.base_game import BaseGame

P1_COLOR = '#e63946'   # red  — always Player 1 (X)
P2_COLOR = '#457b9d'   # blue — always Player 2 (O)
P1_LABEL = 'Player 1 (X)'
P2_LABEL = 'Player 2 (O)'

def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_ttt(board_tuple, title='', path=None, p1_label=None, p2_label=None):
    board = np.array(board_tuple).reshape(3, 3)
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(1, 3):
        ax.axhline(i, color='black', lw=2)
        ax.axvline(i, color='black', lw=2)

    for r in range(3):
        for c in range(3):
            v = board[r, c]
            if v == BaseGame.PLAYER1:
                ax.text(c + 0.5, 2.5 - r, 'X', ha='center', va='center',
                        fontsize=28, fontweight='bold', color=P1_COLOR)
            elif v == BaseGame.PLAYER2:
                ax.text(c + 0.5, 2.5 - r, 'O', ha='center', va='center',
                        fontsize=28, fontweight='bold', color=P2_COLOR)

    legend = [
        Line2D([0], [0], marker='$X$', color='w', markerfacecolor=P1_COLOR,
               markersize=12, label=p1_label or P1_LABEL),
        Line2D([0], [0], marker='$O$', color='w', markerfacecolor=P2_COLOR,
               markersize=12, label=p2_label or P2_LABEL),
    ]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=8, frameon=True)

    if title:
        ax.set_title(title, fontsize=9, pad=4)

    fig.tight_layout()
    _save(fig, path)


def render_ttt_sequence(states, moves, title='', path=None, p1_label=None, p2_label=None):
    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    for idx, (ax, state, move) in enumerate(zip(axes, states, moves)):
        board = np.array(state).reshape(3, 3)
        ax.set_xlim(0, 3); ax.set_ylim(0, 3)
        ax.set_aspect('equal'); ax.axis('off')
        for i in range(1, 3):
            ax.axhline(i, color='black', lw=1.5)
            ax.axvline(i, color='black', lw=1.5)
        for r in range(3):
            for c in range(3):
                v = board[r, c]
                if v == BaseGame.PLAYER1:
                    ax.text(c + 0.5, 2.5 - r, 'X', ha='center', va='center',
                            fontsize=22, fontweight='bold', color=P1_COLOR)
                elif v == BaseGame.PLAYER2:
                    ax.text(c + 0.5, 2.5 - r, 'O', ha='center', va='center',
                            fontsize=22, fontweight='bold', color=P2_COLOR)
        ax.set_title(f'Move {idx+1}: col {move}' if move is not None else 'Start',
                     fontsize=8)

    legend = [
        Line2D([0], [0], marker='$X$', color='w', markerfacecolor=P1_COLOR,
               markersize=10, label=p1_label or P1_LABEL),
        Line2D([0], [0], marker='$O$', color='w', markerfacecolor=P2_COLOR,
               markersize=10, label=p2_label or P2_LABEL),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, path)


def render_connect4(board_tuple, title='', path=None, p1_label=None, p2_label=None):
    from games.connect4 import ROWS, COLS
    board = np.array(board_tuple).reshape(ROWS, COLS)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('#1d3557')
    fig.patch.set_facecolor('#1d3557')

    hole_color = '#f1faee'
    for r in range(ROWS):
        for c in range(COLS):
            v = board[r, c]
            color = P1_COLOR if v == BaseGame.PLAYER1 else (P2_COLOR if v == BaseGame.PLAYER2 else hole_color)
            circ = patches.Circle((c + 0.5, ROWS - r - 0.5), 0.42, color=color, zorder=2)
            ax.add_patch(circ)

    for c in range(COLS):
        ax.text(c + 0.5, -0.3, str(c), ha='center', va='center', fontsize=9, color='white')

    legend = [
        patches.Patch(color=P1_COLOR, label=p1_label or P1_LABEL),
        patches.Patch(color=P2_COLOR, label=p2_label or P2_LABEL),
        patches.Patch(color=hole_color, label='Empty'),
    ]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.06),
              ncol=3, fontsize=9, frameon=True,
              facecolor='#1d3557', labelcolor='white', edgecolor='white')

    if title:
        ax.set_title(title, fontsize=10, color='white', pad=6)

    fig.tight_layout()
    _save(fig, path)


def render_connect4_sequence(states, moves, title='', path=None, p1_label=None, p2_label=None):
    from games.connect4 import ROWS, COLS
    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('#1d3557')

    hole_color = '#f1faee'
    for idx, (ax, state, move) in enumerate(zip(axes, states, moves)):
        board = np.array(state).reshape(ROWS, COLS)
        ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_facecolor('#1d3557')
        for r in range(ROWS):
            for c in range(COLS):
                v = board[r, c]
                color = P1_COLOR if v == BaseGame.PLAYER1 else (P2_COLOR if v == BaseGame.PLAYER2 else hole_color)
                circ = patches.Circle((c + 0.5, ROWS - r - 0.5), 0.38, color=color, zorder=2)
                ax.add_patch(circ)
        label = f'Move {idx*2+1}: col {move}' if move is not None else 'Start'
        ax.set_title(label, fontsize=8, color='white')

    legend = [
        patches.Patch(color=P1_COLOR, label=p1_label or P1_LABEL),
        patches.Patch(color=P2_COLOR, label=p2_label or P2_LABEL),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), facecolor='#1d3557',
               labelcolor='white', edgecolor='white')
    if title:
        fig.suptitle(title, fontsize=10, color='white', y=1.02)
    fig.tight_layout()
    _save(fig, path)


def plot_dqn_loss_curve(loss_history, label, path=None, window=200):
    if not loss_history:
        return
    steps  = np.array([s for s, _ in loss_history])
    losses = np.array([l for _, l in loss_history])
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode='valid')
    x = steps[window - 1:]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, losses, alpha=0.2, color='#e76f51', lw=0.5, label='Raw loss')
    ax.plot(x, smoothed, color='#e76f51', lw=1.5, label=f'Smoothed (window={window})')
    ax.set_xlabel('Training step'); ax.set_ylabel('MSE Loss')
    ax.set_title(f'DQN Training Loss — {label}')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_training_curves_combined(curves, path=None, window=500):
    fig, ax = plt.subplots(figsize=(10, 5))
    kernel = np.ones(window) / window
    for label, results_list in curves:
        wins = np.array([1.0 if r == 'win' else 0.0 for r in results_list])
        smoothed = np.convolve(wins, kernel, mode='valid')
        x = np.arange(window, len(wins) + 1)
        ax.plot(x, smoothed * 100, lw=1.5, label=label)
    ax.set_xlabel('Episode'); ax.set_ylabel('Win rate (%)')
    ax.set_title(f'RL Training Curves — Win Rate (smoothed, window={window})')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_training_curves_detailed(label, results_list, path=None, window=500):
    wins   = np.array([1.0 if r == 'win'  else 0.0 for r in results_list])
    draws  = np.array([1.0 if r == 'draw' else 0.0 for r in results_list])
    losses = np.array([1.0 if r == 'loss' else 0.0 for r in results_list])
    kernel = np.ones(window) / window
    x = np.arange(window, len(wins) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, np.convolve(wins,   kernel, 'valid') * 100, color='#2a9d8f', lw=1.5, label='Win')
    ax.plot(x, np.convolve(draws,  kernel, 'valid') * 100, color='#e9c46a', lw=1.5, label='Draw')
    ax.plot(x, np.convolve(losses, kernel, 'valid') * 100, color='#e76f51', lw=1.5, label='Loss')
    ax.set_xlabel('Episode'); ax.set_ylabel('Rate (%)')
    ax.set_title(f'{label} — Win/Draw/Loss over Training (window={window})')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_vs_default(agent_names, win_rates, draw_rates, loss_rates, title, path=None):
    x = np.arange(len(agent_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, win_rates,  w, label='Win',  color='#2a9d8f')
    ax.bar(x,     draw_rates, w, label='Draw', color='#e9c46a')
    ax.bar(x + w, loss_rates, w, label='Loss', color='#e76f51')
    ax.set_xticks(x); ax.set_xticklabels(agent_names, rotation=15, ha='right')
    ax.set_ylabel('Rate (%)'); ax.set_ylim(0, 110); ax.set_title(title)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.1f%%', fontsize=7, padding=2)
    fig.tight_layout()
    _save(fig, path)


def plot_head_to_head_matrix(agent_names, win_matrix, title, path=None):
    n = len(agent_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    masked = np.ma.masked_where(np.eye(n, dtype=bool), win_matrix)
    im = ax.imshow(masked, vmin=0, vmax=100, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='Win rate (%)')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(agent_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(agent_names, fontsize=9)
    ax.set_title(title); ax.set_xlabel('Opponent'); ax.set_ylabel('Agent')
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f'{win_matrix[i, j]:.0f}%', ha='center', va='center',
                        fontsize=9, color='black' if 30 < win_matrix[i, j] < 70 else 'white')
    fig.tight_layout()
    _save(fig, path)


def plot_game_length_distribution(length_data, title, path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    for idx, (label, lengths) in enumerate(length_data.items()):
        ax.hist(lengths, bins=range(min(lengths), max(lengths)+2),
                alpha=0.6, label=label, color=colors[idx % 10], density=True)
    ax.set_xlabel('Game length (moves)'); ax.set_ylabel('Density')
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_first_mover_advantage(agent_names, p1_wins, p2_wins, draws, title, path=None):
    x = np.arange(len(agent_names))
    w = 0.5
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, p1_wins, w, label='P1 wins (goes first)', color='#2a9d8f')
    ax.bar(x, draws,   w, bottom=p1_wins, label='Draw', color='#e9c46a')
    ax.bar(x, p2_wins, w, bottom=np.array(p1_wins) + np.array(draws),
           label='P2 wins (goes second)', color='#e76f51')
    ax.set_xticks(x); ax.set_xticklabels(agent_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Rate (%)'); ax.set_ylim(0, 110); ax.set_title(title)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_hyperparam_sensitivity(param_name, param_values, results_dict, title, path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, win_rates in results_dict.items():
        ax.plot(range(len(param_values)), win_rates, 'o-', lw=1.5, label=label)
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_xlabel(param_name); ax.set_ylabel('Win rate (%)')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_hyperparam_heatmap(param1_name, param1_vals, param2_name, param2_vals,
                             win_matrix, title, path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(win_matrix, vmin=0, vmax=100, cmap='YlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='Win rate (%)')
    ax.set_xticks(range(len(param2_vals))); ax.set_yticks(range(len(param1_vals)))
    ax.set_xticklabels([str(v) for v in param2_vals])
    ax.set_yticklabels([str(v) for v in param1_vals])
    ax.set_xlabel(param2_name); ax.set_ylabel(param1_name); ax.set_title(title)
    for i in range(len(param1_vals)):
        for j in range(len(param2_vals)):
            ax.text(j, i, f'{win_matrix[i, j]:.0f}%', ha='center', va='center', fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def plot_ttt_qvalue_heatmap(q_table, player, title, path=None):
    empty_state = (0,) * 9
    norm_state = tuple(v * player for v in empty_state)
    q_vals = np.zeros(9)
    if norm_state in q_table:
        for a in range(9):
            q_vals[a] = q_table[norm_state][a]
    board = q_vals.reshape(3, 3)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(board, cmap='RdYlGn', aspect='equal')
    plt.colorbar(im, ax=ax, label='Q-value')
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f'{board[r,c]:.2f}', ha='center', va='center', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    for i in [0.5, 1.5]:
        ax.axhline(i, color='black', lw=1)
        ax.axvline(i, color='black', lw=1)
    fig.tight_layout()
    _save(fig, path)
