# Configuration file for MuZero Knockoff - PROFESSOR OPTIMIZED VERSION

# Game Parameters (BitFall)
GRID_WIDTH = 10
GRID_HEIGHT = 15
STATE_CHANNELS = 3      # One-hot channels: Empty, Debris, Receptor
INITIAL_RECEPTOR_SIZE = 3
INITIAL_RECEPTOR_OFFSET = 0
INITIAL_RECEPTORS = 1   # One contiguous starting receptor segment in this BitFall setup
RECEPTOR_SIZE = INITIAL_RECEPTOR_SIZE  # Backward-compatible alias used by older artifacts
MAX_EPISODE_STEPS = 100
DEBRIS_SPAWN_THRESHOLD = 0.7

# MuZero Parameters - Optimized for Fast and Effective Training
LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.95
TRAINING_INTERVAL = 2
NUM_EPISODES = 1500     # Higher count is possible due to simulations reduction
MINIBATCH_SIZE = 128
BUFFER_CAPACITY = 2000

# u-MCTS Parameters - High Efficiency (as suggested by Professor)
MCTS_SIMULATIONS = 50   # DeepMind used 50 for Atari; sufficient for BitFall
MAX_SEARCH_DEPTH = 6
PUCT_C = 1.5

# BPTT Parameters
LOOK_BACK = 1           # Full state-window size |phi_k| = q + 1. For BitFall we use q = 0.
ROLL_AHEAD = 6          # Imagination window (w)

# Neural Network Parameters
ABSTRACT_STATE_DIM = 128
POLICY_DIM = 3
HIDDEN_LAYER_DIM = 256
WEIGHT_INIT_SCALE = 1e-2
WEIGHTS_PATH = "muzero_weights_optimized.pkl"  # Saved inside each run_* directory

# Loss Function Weights
LOSS_W_POLICY = 1.0
LOSS_W_VALUE = 0.25
LOSS_W_REWARD = 1.0

# Visualization
RENDER_GAME = False
SHOW_PLOTS = False  # Keep training reproducible in headless environments; plots are still saved to disk.
PLOT_PAUSE_DURATION = 0.01
GLOBAL_SEED = 42
EVAL_EPISODES = 50
RUN_VERIFICATION_CHECKS = True
