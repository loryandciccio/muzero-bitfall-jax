import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "project2it3105-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_PAUSE_DURATION

class LivePlotter:
    """Class to update training plots in real-time."""
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.line1, = self.ax1.plot([], [], label="Loss")
        self.line2, = self.ax2.plot([], [], label="Avg Reward (last 10)", color='orange')
        
        self.ax1.set_title("Training Loss")
        self.ax1.set_xlabel("Training Steps")
        self.ax1.set_ylabel("Loss")
        
        self.ax2.set_title("Total Rewards per Episode")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylabel("Total Reward")
        
        self.losses = []
        self.rewards = []
        self.avg_rewards = []

    def update(self, losses, rewards):
        """Update the plot with new data."""
        self.losses = losses
        self.rewards = rewards
        
        # Calculate moving average for rewards
        if len(rewards) >= 10:
            avg = np.mean(rewards[-10:])
        else:
            avg = np.mean(rewards) if rewards else 0
        self.avg_rewards.append(avg)

        # Update Loss plot
        self.line1.set_data(range(len(self.losses)), self.losses)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update Reward plot (showing moving average)
        self.line2.set_data(range(len(self.avg_rewards)), self.avg_rewards)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(PLOT_PAUSE_DURATION)

def plot_training_results(losses, rewards, save_path=None, show=True):
    """Static plot of loss and total rewards over episodes."""
    plt.ioff()
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if losses:
        plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    if rewards:
        plt.plot(rewards)
    plt.title("Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.tight_layout()
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        except IOError as e:
            print(f"Warning: Could not save plot to {save_path}: {e}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

def visualize_state(state):
    """
    Visualize the BitFall grid state in a matplotlib window.
    
    Uses a persistent figure to avoid flickering during sequential updates.
    Grid values: 0=empty (dark), 1=debris (yellow), 2=receptor (purple).
    
    Args:
        state: numpy array of shape (GRID_HEIGHT, GRID_WIDTH) with values 0/1/2
    """
    # Use a persistent figure to avoid flickering and slow downs
    if not hasattr(visualize_state, "fig"):
        plt.ion()
        visualize_state.fig, visualize_state.ax = plt.subplots()
        visualize_state.im = visualize_state.ax.imshow(state, cmap='viridis')
        visualize_state.ax.set_title("BitFall Game State")
    else:
        visualize_state.im.set_data(state)
        
    visualize_state.fig.canvas.draw()
    visualize_state.fig.canvas.flush_events()
    plt.pause(0.05)
