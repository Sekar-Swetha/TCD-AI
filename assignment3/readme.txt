CS7IS2 Assignment 3 — README
Name: Swetha Sekar
Student ID: 25336453

DEPENDENCIES:

Python 3.9+
  pip3 install torch numpy matplotlib pandas

All commands are run from the project root directory (ai2/).


QUICK START:

To run the entire pipeline (training, evaluation, curriculum, P1/P2 analysis):

  ./run.sh

Steps run in order:
  1. Hyperparameter search (TTT)
  2. Train & evaluate TTT agents
  3. Train & evaluate Connect 4 agents
  4. Curriculum training (TTT + C4)
  5. Curriculum evaluation
  6. P1 vs P2 first-mover analysis

Results saved to results/  and models/


TRAINING RL AGENTS:

Individual training commands:

  python3 main.py --game ttt      --train ql    # Q-Learning TTT  (~1 min,  50k episodes vs default)
  python3 main.py --game ttt      --train dqn   # DQN TTT         (~4 min,  50k episodes vs default)
  python3 main.py --game connect4 --train ql    # Q-Learning C4   (~2 min, 200k episodes vs random)
  python3 main.py --game connect4 --train dqn   # DQN C4          (~13 min, 50k episodes vs random)

Pre-trained models included in models/:
  models/ttt_ql.pkl              Tabular Q-Learning for TTT        (~17 KB)
  models/ttt_dqn.pt              DQN for TTT                       (~305 KB)
  models/c4_dqn.pt               DQN for Connect 4                 (~708 KB)
  models/ttt_ql_curriculum.pkl   Curriculum Q-Learning for TTT
  models/ttt_dqn_curriculum.pt   Curriculum DQN for TTT
  models/c4_ql_curriculum.pkl    Curriculum Q-Learning for C4
  models/c4_dqn_curriculum.pt    Curriculum DQN for C4

NOTE: models/c4_ql.pkl (~345 MB, 716k states) exceeds the submission size limit
and is NOT included. Regenerate with:
  python3 main.py --game connect4 --train ql


RUNNING EVALUATIONS:

Full evaluation suites (trains if needed, saves all charts):
  python3 training/train_ttt.py
  python3 training/train_connect4.py

Curriculum training and evaluation:
  python3 training/curriculum_training.py --game both
  python3 training/evaluate_curriculum.py --game both

P1 vs P2 first-mover analysis:
  python3 training/gen_p1p2_analysis.py

Via main.py:
  python3 main.py --game ttt      --eval all
  python3 main.py --game connect4 --eval all



PLAYING SPECIFIC MATCHUPS:

Agent names: minimax, minimax_ab, ql, dqn, default, random, human

Tic Tac Toe:
  python3 main.py --game ttt --agent minimax    --opponent default   --games 100
  python3 main.py --game ttt --agent minimax_ab --opponent default   --games 100
  python3 main.py --game ttt --agent ql         --opponent default   --games 500
  python3 main.py --game ttt --agent dqn        --opponent default   --games 500
  python3 main.py --game ttt --agent minimax_ab --opponent ql        --games 200
  python3 main.py --game ttt --agent minimax_ab --opponent dqn       --games 200
  python3 main.py --game ttt --agent ql         --opponent dqn       --games 200
  python3 main.py --game ttt --agent human      --opponent minimax_ab

Connect 4:
  python3 main.py --game connect4 --agent minimax_ab --opponent random   --games 100
  python3 main.py --game connect4 --agent ql         --opponent random   --games 200
  python3 main.py --game connect4 --agent dqn        --opponent random   --games 200
  python3 main.py --game connect4 --agent minimax_ab --opponent ql       --games 100
  python3 main.py --game connect4 --agent minimax_ab --opponent dqn      --games 100
  python3 main.py --game connect4 --agent ql         --opponent dqn      --games 100
  python3 main.py --game connect4 --agent human      --opponent dqn



OUTPUT / RESULTS:

results/ttt/
  training_curves.png       QL and DQN win rate during training
  curriculum_curves.png     Curriculum training progression
  minimax_comparison.png    Plain vs alpha-beta nodes visited and time
  vs_default.png            All agents vs default opponent
  head_to_head.png          Agent vs agent win rate matrix
  p1_vs_p2.png              First-mover win rates by matchup
  game_*.png                Board snapshots and move sequences

results/connect4/
  training_curves.png       QL and DQN win rate during training
  curriculum_curves.png     Curriculum training progression
  dqn_loss_curriculum.png   DQN Bellman loss during curriculum training
  minimax_scalability.png   Nodes and time vs depth limit
  vs_random.png             All agents vs random opponent
  head_to_head.png          Agent vs agent win rate matrix
  p1_vs_p2.png              First-mover win rates by matchup
  game_*.png                Board snapshots and move sequences



PROJECT STRUCTURE:

ai2/
  games/
    base_game.py             Abstract game interface
    tictactoe.py             Tic Tac Toe (3x3)
    connect4.py              Connect 4 (6x7)
    opponent.py              Default semi-intelligent opponent
  agents/
    base_agent.py            Abstract agent interface
    minimax.py               Minimax (plain + alpha-beta) + C4 heuristic
    q_learning.py            Tabular Q-Learning
    dqn.py                   Deep Q-Network (PyTorch)
  training/
    evaluate.py              Tournament runner (library, not a script)
    train_ttt.py             Full TTT training + evaluation pipeline
    train_connect4.py        Full C4 training + evaluation pipeline
    curriculum_training.py   Curriculum training (Random -> Default -> Minimax)
    evaluate_curriculum.py   Curriculum agent evaluation vs all opponents
    gen_p1p2_analysis.py     First-mover (P1 vs P2) bias analysis
    hyperparameter_search.py Grid search for QL and DQN hyperparameters
    analysis.py              Minimax timing and scalability analysis
    results_logger.py        CSV result logging utilities
  models/                    Saved model weights
  results/                   Generated charts and CSV summaries
  visualizer.py              Matplotlib board renderer and chart generators
  config.py                  All hyperparameters
  main.py                    Unified CLI entry point
  run.sh                     Runs the full pipeline end to end
  readme.txt                 This file
