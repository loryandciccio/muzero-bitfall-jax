import numpy as np
from config import LOOK_BACK, STATE_CHANNELS

def to_code(state_sequence):
    """
    Converts a sequence of states to an encoded format (Code) for NN input.
    Uses one-hot encoding for the possible values in the domain.
    In this codebase LOOK_BACK denotes the full state-window size, so the
    MuZero pseudocode parameter q corresponds to LOOK_BACK - 1.
    Input: List of numpy arrays [s_k-q, ..., s_k], each (H, W).
    Output: JAX-compatible array of shape (H, W, STATE_CHANNELS, LOOK_BACK)
            containing one-hot encoded channels for each state.
    """
    if len(state_sequence) < LOOK_BACK:
        blank = np.zeros_like(state_sequence[0])
        state_sequence = [blank for _ in range(LOOK_BACK - len(state_sequence))] + list(state_sequence)
    elif len(state_sequence) > LOOK_BACK:
        state_sequence = state_sequence[-LOOK_BACK:]

    one_hots = []
    for s in state_sequence:
        s = np.asarray(s, dtype=int)
        s = np.clip(s, 0, STATE_CHANNELS - 1)
        oh = np.eye(STATE_CHANNELS)[s]  # Shape (H, W, STATE_CHANNELS)
        one_hots.append(oh)
    
    # Final stack shape (H, W, STATE_CHANNELS, LOOK_BACK)
    return np.stack(one_hots, axis=-1).astype(np.float32)

def get_blank_state(width, height):
    """Returns a blank state grid."""
    return np.zeros((height, width), dtype=int)
