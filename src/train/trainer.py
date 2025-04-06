from typing import Dict, List, Optional

import numpy as np

from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent, Experience
from src.utils.visualization import TradingVisualizer


class StockTrader:
    """
    A class to manage the training of the DQN agent for stock trading.

    Attributes:
        agent (DQNAgent): The DQN agent that learns to trade
        env (StockTradingEnv): The trading environment
        visualizer (TradingVisualizer): Visualization tool for training progress
        history (dict): Dictionary to store training metrics
    """

    def __init__(self, env: StockTradingEnv, agent: Optional[DQNAgent] = None):
        """
        Initialize the stock trader.

        Args:
            env: The trading environment
            agent: Optional pre-trained agent. If None, a new agent will be created.
        """
        self.env = env
        # Handle potentially None shape safely
        state_size = (
            env.observation_space.shape[0] if env.observation_space.shape is not None else 0
        )
        # Use getattr to safely get the 'n' attribute or default to 0
        action_size = getattr(env.action_space, "n", 0)

        self.agent = agent if agent else DQNAgent(state_size, action_size)
        self.visualizer = TradingVisualizer()
        self.history: Dict[str, List[float]] = {"episode_rewards": [], "portfolio_values": []}

    def train(
        self, episodes: int, batch_size: int = 32, render_interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the agent on the environment.

        Args:
            episodes: Number of episodes to train
            batch_size: Size of batches for experience replay
            render_interval: Interval at which to render and save visualizations

        Returns:
            Dictionary containing training history
        """
        print(f"Starting training for {episodes} episodes...")

        for episode in range(episodes):
            # Unpack tuple returned by reset
            reset_result = self.env.reset()
            state = reset_result[0]  # Extract observation
            total_reward = 0.0
            done = False

            while not done:
                # Agent selects action
                action = self.agent.act(state)

                # Take action in environment and handle 5 return values
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Combine termination flags
                done = terminated or truncated

                # Store experience in replay memory
                self.agent.remember(
                    Experience(
                        state=state, action=action, reward=reward, next_state=next_state, done=done
                    )
                )

                # Move to next state
                state = next_state
                total_reward += reward

                # Train agent on a batch of experiences
                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)

            # Update target model periodically
            if episode % 10 == 0:
                self.agent.update_target_model()

            # Store episode metrics
            self.history["episode_rewards"].append(total_reward)
            final_price = self.env.df["Close"].iloc[-1]
            portfolio_value = self.env.balance + self.env.shares_held * final_price
            self.history["portfolio_values"].append(portfolio_value)

            # Print progress
            if episode % render_interval == 0:
                avg_reward = np.mean(self.history["episode_rewards"][-render_interval:])
                print(f"Episode: {episode}/{episodes}")
                print(f"Average Reward (last {render_interval} episodes): {avg_reward:.2f}")
                print(f"Portfolio Value: ${self.history['portfolio_values'][-1]:.2f}")
                print(f"Epsilon: {self.agent.epsilon:.4f}")
                print("-" * 50)

                # Update visualizations
                self.visualizer.plot_training_history(
                    self.history, save_path="results/training_progress.html"
                )

        print("Training completed!")
        return self.history

    def save_agent(self, filepath: str) -> None:
        """
        Save the trained agent's model weights.

        Args:
            filepath: Path to save the model weights
        """
        # Fix: use the save method instead of model.save_weights
        self.agent.save(filepath)
        print(f"Agent saved to {filepath}")

    def load_agent(self, filepath: str) -> None:
        """
        Load pre-trained agent weights.

        Args:
            filepath: Path to the saved model weights
        """
        # Fix: use the load method instead of model.load_weights
        self.agent.load(filepath)
        print(f"Agent loaded from {filepath}")
