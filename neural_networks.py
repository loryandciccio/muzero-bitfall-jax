import jax
import jax.numpy as jnp
from jax import random
import os
from config import (
    ABSTRACT_STATE_DIM,
    POLICY_DIM,
    LOOK_BACK,
    GRID_WIDTH,
    GRID_HEIGHT,
    HIDDEN_LAYER_DIM,
    WEIGHT_INIT_SCALE,
    STATE_CHANNELS,
)

# Helper to initialize weights
def random_layer_params(m, n, key, scale=WEIGHT_INIT_SCALE):
    """
    Initialize weight matrix and bias vector for a single layer.
    
    Args:
        m: Input dimension
        n: Output dimension
        key: JAX random key
        scale: Scaling factor for random initialization
        
    Returns:
        Tuple of (weights, biases) as JAX arrays
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
    """
    Initialize parameters for a multi-layer perceptron.
    
    Args:
        sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        key: JAX random key
        
    Returns:
        List of (weight, bias) tuples for each layer
    """
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
    """
    ReLU activation function.
    
    Args:
        x: Input array
        
    Returns:
        max(0, x) element-wise
    """
    return jnp.maximum(0, x)

def softmax(x):
    """
    Softmax activation function for probability distributions.
    
    Args:
        x: Input logits
        
    Returns:
        Normalized probabilities summing to 1
    """
    return jax.nn.softmax(x)

def predict_mlp(params, x):
    """
    Forward pass through a multi-layer perceptron.
    
    Args:
        params: List of (weight, bias) tuples for each layer
        x: Input vector
        
    Returns:
        Output vector after all layers (ReLU on hidden, linear on output)
    """
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    final_w, final_b = params[-1]
    return jnp.dot(final_w, activations) + final_b

import pickle

class MuZeroNet:
    """Manager for the three MuZero networks."""
    
    def __init__(self, seed=42):
        self.key = random.PRNGKey(seed)
        self.key, repr_key, dyna_key, pred_key = random.split(self.key, 4)
        
        # Representation Network: Grid (W*H*q*CHANNELS) -> Abstract State (dim)
        input_dim = GRID_WIDTH * GRID_HEIGHT * LOOK_BACK * STATE_CHANNELS
        self.repr_params = init_network_params([input_dim, HIDDEN_LAYER_DIM, ABSTRACT_STATE_DIM], repr_key)
        
        # Dynamics Network: (Abstract State + Action) -> (Next Abstract State, Reward)
        # Action is one-hot encoded (dim 3)
        self.dyna_params = init_network_params([ABSTRACT_STATE_DIM + POLICY_DIM, HIDDEN_LAYER_DIM, ABSTRACT_STATE_DIM + 1], dyna_key)
        
        # Prediction Network: Abstract State -> (Policy Logits, Value)
        self.pred_params = init_network_params([ABSTRACT_STATE_DIM, HIDDEN_LAYER_DIM, POLICY_DIM + 1], pred_key)

    def save_params(self, params, path):
        """Save parameters to a file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
            with open(path, "wb") as f:
                pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Parameters saved to {path}")
        except IOError as e:
            raise IOError(f"Failed to save parameters to {path}: {e}")

    def load_params(self, path):
        """Load parameters from a file."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file not found: {path}")
        except (IOError, pickle.UnpicklingError) as e:
            raise IOError(f"Failed to load parameters from {path}: {e}")

    def representation(self, params, states_code):
        x = jnp.asarray(states_code, dtype=jnp.float32).flatten()
        return relu(predict_mlp(params, x))

    def dynamics(self, params, abstract_state, action):
        # action is an int 0, 1, 2
        one_hot_action = jax.nn.one_hot(action, POLICY_DIM)
        x = jnp.concatenate([jnp.asarray(abstract_state, dtype=jnp.float32), one_hot_action])
        out = predict_mlp(params, x)
        next_state = relu(out[:ABSTRACT_STATE_DIM])
        reward = out[-1]
        return next_state, reward

    def prediction(self, params, abstract_state):
        out = predict_mlp(params, abstract_state)
        policy_logits = out[:POLICY_DIM]
        value = out[-1]  # Remove tanh to allow linear value range
        return policy_logits, value

    def get_all_params(self):
        return {
            'repr': self.repr_params,
            'dyna': self.dyna_params,
            'pred': self.pred_params
        }
