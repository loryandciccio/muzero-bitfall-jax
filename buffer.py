import random
import numpy as np
from config import LOOK_BACK, ROLL_AHEAD, DISCOUNT_FACTOR, POLICY_DIM

class EpisodeBuffer:
    """Buffer to store and sample episodic data for MuZero training."""
    
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.episodes = []

    def save_episode(self, episode_data):
        """
        Store a complete episode in the buffer.
        
        When buffer reaches capacity, removes oldest episode (FIFO).
        
        Args:
            episode_data: List of [state, value, policy, action, reward] tuples
        """
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode_data)

    def sample_batch(self, batch_size):
        """
        Sample a batch of BPTT training sequences from stored episodes.
        
        For each sample:
        - Selects random episode and starting position k
        - Gathers LOOK_BACK states (q+1 window) with zero-padding if needed
        - Gathers ROLL_AHEAD actions and rewards (w steps)
        - Gathers ROLL_AHEAD+1 policy and value targets
        - Computes bootstrapped target values with discounting
        
        Args:
            batch_size: Number of sequences to sample
            
        Returns:
            List of dicts with keys: 'states', 'actions', 'target_policies', 
            'target_values', 'target_rewards'
            
        Raises:
            RuntimeError: If buffer contains no valid episodes
        """
        if batch_size <= 0:
            return []

        valid_episodes = [ep for ep in self.episodes if len(ep) > 0]
        if not valid_episodes:
            raise RuntimeError("Cannot sample from empty buffer. No valid episodes available.")

        uniform_policy = np.ones(POLICY_DIM, dtype=np.float32) / POLICY_DIM
        batch = []
        for _ in range(batch_size):
            episode = random.choice(valid_episodes)
            k = random.randint(0, len(episode) - 1)
            
            # Gather the state window phi_k. In this codebase LOOK_BACK is the
            # full window size (q + 1 in the project pseudocode).
            states = []
            for i in range(k - LOOK_BACK + 1, k + 1):
                if i < 0:
                    states.append(np.zeros_like(episode[0][0]))
                else:
                    states.append(episode[min(i, len(episode) - 1)][0])
            
            # Gather future actions, policies, values, rewards for w steps with safe padding
            actions = []
            target_policies = []
            target_rewards = []

            for i in range(k, k + ROLL_AHEAD):
                if i < len(episode):
                    actions.append(int(episode[i][3]))
                    target_rewards.append(float(episode[i][4]))
                else:
                    actions.append(0)
                    target_rewards.append(0.0)

            for i in range(k, k + ROLL_AHEAD + 1):
                if i < len(episode):
                    target_policies.append(np.asarray(episode[i][2], dtype=np.float32))
                else:
                    target_policies.append(uniform_policy)
            
            # Calculate target values (bootstrapped return)
            target_values = []
            for i in range(k, k + ROLL_AHEAD + 1):
                z = self._calculate_target_value(episode, i)
                target_values.append(z)

            batch.append({
                'states': states,
                'actions': actions,
                'target_policies': target_policies,
                'target_values': target_values,
                'target_rewards': target_rewards
            })
        return batch

    def _calculate_target_value(self, episode, start_idx):
        """
        Calculate the n-step bootstrapped return (z_k) used as the value target.
        
        Temporal mapping in episode_data:
        - episode[i][1] -> MCTS value estimate for state s_i (v_i)
        - episode[i][4] -> Real reward obtained after executing action a_{i+1} (r_{i+1})
        
        The target z_k is computed as the discounted sum of rewards over n steps
        (where n = ROLL_AHEAD) plus the bootstrapped value at the horizon when
        that horizon still lies inside the episode:
        z_k = r_{k+1} + gamma*r_{k+2} + ... + gamma^{n-1}*r_{k+n} + gamma^n * v_{k+n}

        If the rollout reaches the end of the episode before n steps, the
        remaining bootstrap term is omitted and we use only the discounted
        rewards that are actually available.
        """
        if start_idx >= len(episode):
            return 0.0

        z = 0.0
        gamma = 1.0
        reward_end = min(start_idx + ROLL_AHEAD, len(episode))

        # Accumulate discounted rewards from r_{k+1} onward, stopping at the
        # episode boundary if fewer than ROLL_AHEAD rewards are available.
        for i in range(start_idx, reward_end):
            z += gamma * episode[i][4]
            gamma *= DISCOUNT_FACTOR

        # Add the bootstrapped MCTS value only when the horizon state exists.
        if reward_end < len(episode):
            z += gamma * episode[reward_end][1]
        return float(z)
