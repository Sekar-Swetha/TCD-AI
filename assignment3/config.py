MINIMAX_DEPTH_TTT = None   # full search (tree is tiny)
MINIMAX_DEPTH_C4  = 3      # depth-limited for Connect 4 (0.032s/move w/ AB; depth-5 is 0.57s)

QL_ALPHA         = 0.1
QL_GAMMA         = 0.95
QL_EPSILON       = 1.0
QL_EPSILON_MIN   = 0.05
QL_EPSILON_DECAY = 0.9995
QL_EPISODES_TTT  = 50_000
QL_EPISODES_C4   = 200_000   # fast — ~2 min

DQN_LR              = 1e-3
DQN_GAMMA           = 0.95
DQN_EPSILON         = 1.0
DQN_EPSILON_MIN     = 0.05
DQN_EPSILON_DECAY   = 0.9995
DQN_BATCH_SIZE      = 64
DQN_BUFFER_CAPACITY = 50_000
DQN_TARGET_UPDATE   = 500
DQN_EPISODES_TTT    = 50_000
DQN_EPISODES_C4     = 50_000    # ~13 min on CPU

# TTT network: input=9 cells, output=9 moves
DQN_INPUT_TTT   = 9
DQN_OUTPUT_TTT  = 9
DQN_HIDDEN_TTT  = (128, 128)

# C4 network: input=42 cells, output=7 columns
DQN_INPUT_C4    = 42
DQN_OUTPUT_C4   = 7
DQN_HIDDEN_C4   = (256, 128)

EVAL_GAMES     = 500   # games per matchup for TTT
EVAL_GAMES_C4  = 200   # games per matchup for Connect 4 (minimax is slower)

MODEL_DIR = 'models'
RESULTS_DIR = 'results'
