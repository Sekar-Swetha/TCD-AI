from games.base_game import BaseGame


def run_tournament(agent1, agent2, game, n_games=500, swap_players=True):
    results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0, 'total': n_games}

    for i in range(n_games):
        game.reset()

        if swap_players and i % 2 == 1:
            a1_role, a2_role = BaseGame.PLAYER2, BaseGame.PLAYER1
        else:
            a1_role, a2_role = BaseGame.PLAYER1, BaseGame.PLAYER2

        _set_player(agent1, a1_role)
        _set_player(agent2, a2_role)

        while not game.is_terminal():
            current = game.get_current_player()
            if current == a1_role:
                move = agent1.select_action(game)
            else:
                move = agent2.select_action(game)
            game.make_move(move)

        winner = game.check_winner()
        if winner == a1_role:
            results['agent1_wins'] += 1
        elif winner == a2_role:
            results['agent2_wins'] += 1
        else:
            results['draws'] += 1

    return results


def _set_player(agent, player):
    if hasattr(agent, 'player'):
        agent.player = player


def print_results(name1, name2, results):
    t = results['total']
    w1 = results['agent1_wins']
    w2 = results['agent2_wins']
    d  = results['draws']
    print(f"  {name1:25s} vs {name2:25s} | "
          f"W1={w1:4d} ({w1/t*100:5.1f}%)  "
          f"W2={w2:4d} ({w2/t*100:5.1f}%)  "
          f"D={d:4d} ({d/t*100:5.1f}%)")
