from simulator import BitFall
from config import POLICY_DIM

class GameStateManager:
    """
    Understands the game rules and state transitions.
    Acts as the Game State Manager (GSM) mentioned in the project specification.
    """
    def __init__(self, seed=None):
        self.simulator = BitFall(seed=seed)

    def produce_initial_state(self):
        """Generates the initial state for a game."""
        return self.simulator.reset()

    def get_legal_actions(self, state=None):
        """Returns all legal actions (Stay, Left, Right)."""
        return list(range(POLICY_DIM))

    def step(self, action):
        """Given a state and action, return next state and reward."""
        return self.simulator.step(action)

    def is_final_state(self):
        """BitFall is continuous, but we define a limit from config."""
        from config import MAX_EPISODE_STEPS
        return self.simulator.step_count >= MAX_EPISODE_STEPS

class AbstractStateManager:
    """
    Maps sequences to abstract states and manages transitions in latent space.
    Acts as the ASM mentioned in the project specification.
    """
    def __init__(self, networks, params):
        self.net = networks
        self.params = params

    def get_abstract_state(self, state_sequence):
        """Map a sequence of game states to a single abstract state."""
        from representations import to_code
        s_code = to_code(state_sequence)
        return self.net.representation(self.params['repr'], s_code)

    def get_predictions(self, abstract_state):
        """Given an abstract state, produce a policy and a predicted value."""
        return self.net.prediction(self.params['pred'], abstract_state)

    def get_next_abstract_state(self, abstract_state, action):
        """Given an abstract state and action, return next abstract state and reward."""
        return self.net.dynamics(self.params['dyna'], abstract_state, action)

    def get_legal_actions(self, abstract_state=None):
        """Normally all possible actions in abstract space."""
        return list(range(POLICY_DIM))
