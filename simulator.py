import numpy as np
import random
from config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    INITIAL_RECEPTOR_OFFSET,
    INITIAL_RECEPTOR_SIZE,
    DEBRIS_SPAWN_THRESHOLD,
)

class BitFall:
    """
    Simulator for the BitFall game.
    Debris (red) falls one row per timestep.
    Receptors (blue) move along the bottom row.
    """

    def __init__(self, seed=None):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        """Reset the game state."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        # The assignment describes an original receptor pattern chosen by the user.
        # In this implementation we support a configurable contiguous starting
        # segment via config: size + offset on the toroidal bottom row.
        self.receptor_offset = int(INITIAL_RECEPTOR_OFFSET) % self.width
        self.receptor_size = int(INITIAL_RECEPTOR_SIZE)
        if self.receptor_size <= 0 or self.receptor_size > self.width:
            raise ValueError(
                f"INITIAL_RECEPTOR_SIZE must be in [1, {self.width}], got {self.receptor_size}."
            )
        self.score = 0
        self.step_count = 0
        self._spawn_debris()
        return self.get_state()

    def _spawn_debris(self):
        """Spawn a random row of debris at the top."""
        new_row = np.zeros(self.width, dtype=int)
        if self.rng.random() < DEBRIS_SPAWN_THRESHOLD:
            start = self.rng.randint(0, self.width - 2)
            length = self.rng.randint(1, min(4, self.width - start))
            new_row[start:start+length] = 1
        self.grid[0] = new_row

    def step(self, action):
        """
        Apply action: 0 (Stay), 1 (Left), 2 (Right)
        Returns: next_state, reward
        """
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}. Expected one of [0, 1, 2].")

        self.step_count += 1

        # 1. Debris falls down first. The row that was at -2 moves to the bottom (-1).
        # The previous bottom row is "removed" by this shift.
        self.grid[1:] = self.grid[:-1]

        # 2. Apply receptor movement.
        self._move_receptors(action)

        # 3. Interaction between the NEW bottom row of debris and the receptors determines reward.
        reward = self._calculate_reward()
        self.score += reward

        # 4. A random row of debris is added to the top at each timestep.
        self._spawn_debris()

        return self.get_state(), reward

    def _move_receptors(self, action):
        """Move receptor row and handle offset."""
        if action == 1: # Left
            self.receptor_offset = (self.receptor_offset - 1) % self.width
        elif action == 2: # Right
            self.receptor_offset = (self.receptor_offset + 1) % self.width

    def _get_receptor_segments(self):
        """
        Calculates receptor segments based on offset and size.
        Implements split-wrap logic as described in the project specification.
        Example: size 3 at offset 9 in width 10 -> indices [9, 0, 1] -> segments [[0, 1], [9]]
        """
        indices = [(self.receptor_offset + i) % self.width for i in range(self.receptor_size)]
        indices.sort()
        
        segments = []
        if not indices: return []
        
        current = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current.append(indices[i])
            else:
                segments.append(current)
                current = [indices[i]]
        segments.append(current)
        return segments

    def _calculate_reward(self):
        """
        Scoring logic (from the project specification):
        'If one segment a) is larger than the other, b) completely covers the other 
        and c) has overlap on one or both sides, then it wins and the number of 
        points achieved equals the size of the losing segment.'
        
        A receptor win entails positive points, while a debris win yields negative points.
        """
        bottom_row = self.grid[-1]
        debris_segments = self._get_debris_segments(bottom_row)
        receptor_segments = self._get_receptor_segments()
        
        reward = 0
        
        for d_seg in debris_segments:
            d_set = set(d_seg)
            for r_seg in receptor_segments:
                r_set = set(r_seg)
                
                # Check for containment and size difference (which implies excess for continuous segments)
                # Receptor wins
                if len(r_seg) > len(d_seg) and d_set.issubset(r_set):
                    reward += len(d_seg)
                # Debris wins
                elif len(d_seg) > len(r_seg) and r_set.issubset(d_set):
                    reward -= len(r_seg)
        
        return reward

    def _get_debris_segments(self, row):
        """Extract continuous segments from a binary row."""
        segments = []
        current = []
        for i, val in enumerate(row):
            if val == 1:
                current.append(i)
            else:
                if current:
                    segments.append(current)
                    current = []
        if current:
            segments.append(current)
        return segments

    def get_state(self):
        """Return the current grid as a numpy array."""
        state_grid = np.copy(self.grid)
        # Clear bottom row from debris (already used for reward) and show receptors
        state_grid[-1] = 0
        receptor_segments = self._get_receptor_segments()
        for segment in receptor_segments:
            for idx in segment:
                state_grid[-1, idx] = 2
        return state_grid

    def render(self):
        """CLI rendering of the game."""
        state = self.get_state()
        print("-" * (self.width + 2))
        for row in state:
            line = "|"
            for cell in row:
                if cell == 1: line += "X"  # Debris
                elif cell == 2: line += "O"  # Receptor
                else: line += " "
            print(line + "|")
        print("-" * (self.width + 2))
        print(f"Score: {self.score} | Step: {self.step_count}")
