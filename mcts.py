import math
import numpy as np
import jax
import jax.numpy as jnp
from config import POLICY_DIM, PUCT_C, MAX_SEARCH_DEPTH, DISCOUNT_FACTOR

class Node:
    """A node in the u-MCTS search tree."""
    def __init__(self, abstract_state, parent=None, action=None, reward=0):
        self.state = abstract_state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.reward = reward  # Predicted reward from Dynamics net
        self.children = {}    # action -> Node
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0      # Probability from Prediction net

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

class uMCTS:
    """
    Specialized Monte Carlo Tree Search for MuZero using abstract states.
    Operates entirely within the latent space, using the Dynamics network 
    to transition between states and the Prediction network for evaluation.
    """
    def __init__(self, asm, rng=None):
        self.asm = asm
        self.rng = rng if rng is not None else np.random.default_rng()

    def search(self, root_state, num_simulations, max_depth, legal_actions=None):
        root = Node(root_state)
        
        # Initial prediction for root to get prior probabilities
        policy_logits, _ = self.asm.get_predictions(root.state)
        root.prior_probs = np.asarray(jax.nn.softmax(policy_logits), dtype=np.float32)

        for _ in range(num_simulations):
            node = root
            search_path = [node]
            depth = 0

            # 1. Selection: Use tree policy to search from root to a leaf (L)
            while node.children and depth < max_depth:
                action, node = self._select_child(node)
                search_path.append(node)
                depth += 1

            # 2. Expansion: ∀ai ∈ A generate child node ci
            if depth < max_depth:
                # Use real legal actions for root if provided
                current_legal = legal_actions if node == root and legal_actions is not None else self.asm.get_legal_actions()
                self._expand_node(node, current_legal)
                
                if not node.children:
                    # No legal moves available
                    _, v_leaf = self.asm.get_predictions(node.state)
                    self._do_backpropagation(search_path, [float(v_leaf)])
                    continue

                # Choose a random child node (c*) of L
                child_actions = list(node.children.keys())
                action_c_star = int(self.rng.choice(child_actions))
                c_star = node.children[action_c_star]
                search_path.append(c_star)
                
                # 3. Rollout: accum reward = DO ROLLOUT(c*, dmax - depth - 1, NNd, NNp)
                accum_rewards = self._do_rollout(c_star.state, max_depth - depth - 1)
                
                # 4. Backpropagation
                self._do_backpropagation(search_path, accum_rewards)
            else:
                # If we already reached max depth during selection, still back up
                # a leaf value so every simulation updates visits and values.
                _, v_leaf = self.asm.get_predictions(node.state)
                self._do_backpropagation(search_path, [float(v_leaf)])

        return root

    def _do_rollout(self, state, depth):
        """
        Implements the DO ROLLOUT logic as described in the project specification.
        Simulates future steps in abstract space using Prediction and Dynamics networks.
        
        Technical Note: In the BitFall domain, all actions (0, 1, 2) are always legal. 
        Therefore, sampling from the predicted policy distribution without an 
        explicit legality mask is both safe and efficient for this knockoff.
        """
        sigma = state
        accum_reward = []
        for _ in range(depth):
            pi_logits, v = self.asm.get_predictions(sigma)
            pi = jax.nn.softmax(pi_logits)
            pi_np = np.asarray(pi, dtype=np.float32)
            action = int(self.rng.choice(self.asm.get_legal_actions(), p=pi_np))
            sigma, r = self.asm.get_next_abstract_state(sigma, action)
            accum_reward.append(float(r))
        
        # Final value prediction
        _, v_final = self.asm.get_predictions(sigma)
        accum_reward.append(float(v_final))
        return accum_reward

    def _do_backpropagation(self, path, rewards_sequence):
        """
        Updates visit counts and Q-values along the path.
        rewards_sequence: [r_c1, r_c2, ..., v_final] - rewards from rollout
        """
        # G is the discounted return from the end of the rollout
        G = 0
        # Start from the end of the rewards sequence (v_final) and work backwards
        # to calculate the discounted return for the last node in the path.
        # But wait, the rewards_sequence corresponds to steps AFTER the last node in path.
        
        # Correct approach for MuZero backup:
        # For a node at depth d, the value is:
        # G = r_{d+1} + gamma * r_{d+2} + ... + gamma^{T-d-1} * r_T + gamma^{T-d} * v_T
        
        # We can compute this iteratively.
        # Let's say path is [n0, n1, ..., nk] where nk is the leaf/c_star.
        # rewards_sequence is the sequence of rewards obtained AFTER nk.
        
        # 1. Start with the value prediction from the end of rollout
        v_final = rewards_sequence[-1]
        rollout_rewards = rewards_sequence[:-1]
        
        # 2. Compute the value at nk (the last node in path)
        G = v_final
        for r in reversed(rollout_rewards):
            G = r + DISCOUNT_FACTOR * G
        
        # 3. Backpropagate up the path
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.visit_count += 1
            node.total_value += G
            
            # For the parent, the return includes the reward on the edge to this node
            if i > 0:
                G = node.reward + DISCOUNT_FACTOR * G

    def _select_child(self, node):
        """
        Select child using PUCT formula.
        Q(s,a) in MuZero is: predicted_reward + gamma * expected_value_of_next_state
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        total_visits = max(1, node.visit_count)

        for action, child in node.children.items():
            # Q-value calculation with discounting and immediate reward
            q_value = child.reward + DISCOUNT_FACTOR * child.value()
            
            # PUCT exploration bonus
            u_value = PUCT_C * node.prior_probs[action] * math.sqrt(total_visits) / (1 + child.visit_count)
            
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand_node(self, node, legal_actions=None):
        """Expand node: ∀ai ∈ A generate child node ci."""
        if legal_actions is None:
            legal_actions = self.asm.get_legal_actions()
            
        # Prediction for priors
        policy_logits, _ = self.asm.get_predictions(node.state)
        node.prior_probs = np.asarray(jax.nn.softmax(policy_logits), dtype=np.float32)

        for a in legal_actions:
            if a not in node.children:
                next_state, reward = self.asm.get_next_abstract_state(node.state, a)
                child = Node(next_state, parent=node, action=a, reward=float(reward))
                node.children[a] = child
