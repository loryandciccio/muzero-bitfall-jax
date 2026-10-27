from functools import partial
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from config import LEARNING_RATE, POLICY_DIM, ROLL_AHEAD, LOSS_W_POLICY, LOSS_W_VALUE, LOSS_W_REWARD
from representations import to_code

class ReinforcementLearningManager:
    """Manager for orchestrating episodes and training MuZero."""
    
    def __init__(self, networks):
        self.net = networks
        self.params = networks.get_all_params()
        self.optimizer = None  # Simple SGD for this knockoff

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch):
        """
        Compute MuZero BPTT loss over a batch of sequences.
        
        Loss components:
        1. Initial prediction loss at timestep k (policy + value)
        2. Unrolled dynamics loss for w steps (reward + policy + value)
        
        All losses use MSE for scalars and cross-entropy for policies.
        
        Args:
            params: Dict with keys 'repr', 'dyna', 'pred' containing network parameters
            batch: Dict of JAX arrays with shapes:
                - states: (batch, H, W, 3, q) one-hot encoded state sequences
                - actions: (batch, w) action indices
                - target_policies: (batch, w+1, 3) MCTS policy distributions
                - target_values: (batch, w+1) bootstrapped value targets
                - target_rewards: (batch, w) observed rewards
                
        Returns:
            Scalar loss averaged over batch
        """
        def single_seq_loss(sample):
            seq_loss = 0.0
            
            # 1. Representation
            s_abstract = self.net.representation(params['repr'], sample['states'])
            
            # Initial prediction
            p_logits, v = self.net.prediction(params['pred'], s_abstract)
            seq_loss += LOSS_W_VALUE * jnp.mean((sample['target_values'][0] - v)**2)
            seq_loss += LOSS_W_POLICY * (-jnp.sum(sample['target_policies'][0] * jax.nn.log_softmax(p_logits)))
            
            # 2. Unroll Dynamics for ROLL_AHEAD imagination steps
            # This follows the 'One Big Elman Net' architecture where gradients
            # flow through consecutive applications of the Dynamics network.
            curr_s = s_abstract
            for i in range(ROLL_AHEAD):
                curr_s, r_pred = self.net.dynamics(params['dyna'], curr_s, sample['actions'][i])
                p_logits, v = self.net.prediction(params['pred'], curr_s)
                
                # Targets
                target_r = sample['target_rewards'][i]
                target_v = sample['target_values'][i+1]
                target_p = sample['target_policies'][i+1]
                
                seq_loss += LOSS_W_REWARD * jnp.mean((target_r - r_pred)**2)
                seq_loss += LOSS_W_VALUE * jnp.mean((target_v - v)**2)
                seq_loss += LOSS_W_POLICY * (-jnp.sum(target_p * jax.nn.log_softmax(p_logits)))
            
            return seq_loss

        # Vectorize over batch
        losses = vmap(single_seq_loss)(batch)
        return jnp.mean(losses)

    def train_step(self, buffer, batch_size):
        """
        Perform one SGD training step on a minibatch sampled from buffer.
        
        Steps:
        1. Sample batch of BPTT sequences from buffer
        2. Convert to JAX arrays with shape validation
        3. Compute loss via backpropagation through time
        4. Calculate gradients with JAX autodiff
        5. Update parameters with simple SGD
        
        Args:
            buffer: EpisodeBuffer instance containing training data
            batch_size: Number of sequences to sample for this update
            
        Returns:
            Float scalar loss value, or None if buffer is empty
        """
        if batch_size <= 0 or not buffer.episodes:
            return None

        try:
            samples = buffer.sample_batch(batch_size)
        except RuntimeError:
            return None

        if not samples:
            return None  # No training performed
        
        # Pre-process batch for JAX
        jax_batch = {
            'states': jnp.asarray([to_code(s['states']) for s in samples], dtype=jnp.float32),
            'actions': jnp.asarray([s['actions'] for s in samples], dtype=jnp.int32),
            'target_policies': jnp.asarray([s['target_policies'] for s in samples], dtype=jnp.float32),
            'target_values': jnp.asarray([s['target_values'] for s in samples], dtype=jnp.float32),
            'target_rewards': jnp.asarray([s['target_rewards'] for s in samples], dtype=jnp.float32),
        }

        # Lightweight shape checks to fail early with clear signal.
        assert jax_batch['states'].ndim == 5
        assert jax_batch['actions'].shape[1] == ROLL_AHEAD
        assert jax_batch['target_rewards'].shape[1] == ROLL_AHEAD
        assert jax_batch['target_policies'].shape[1] == ROLL_AHEAD + 1
        assert jax_batch['target_policies'].shape[2] == POLICY_DIM
        assert jax_batch['target_values'].shape[1] == ROLL_AHEAD + 1

        # Gradient update
        grads = grad(self.loss_fn)(self.params, jax_batch)
        
        # Simple SGD update
        self.params = jax.tree_util.tree_map(
            lambda p, g: p - LEARNING_RATE * g, self.params, grads
        )
        loss = self.loss_fn(self.params, jax_batch)
        return float(loss)

    def play_as_actor(self, state_sequence):
        """
        Deploy trained networks in actor-only mode (no MCTS, no Dynamics).
        
        Uses Representation network to encode state history, then Prediction 
        network to select action greedily (argmax policy).
        
        Args:
            state_sequence: List of LOOK_BACK game states
            
        Returns:
            Integer action (0=Stay, 1=Left, 2=Right)
        """
        s_code = to_code(state_sequence)
        s_abstract = self.net.representation(self.params['repr'], s_code)
        p_logits, _ = self.net.prediction(self.params['pred'], s_abstract)
        # Choose action with highest probability (greedy)
        return int(jnp.argmax(p_logits))
