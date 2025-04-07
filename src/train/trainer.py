import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent, DQNAgentConfig, Experience
from src.utils.visualization import TradingVisualizer

# Configure logger
logger = logging.getLogger(__name__)

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

        if agent is None:
            # Create agent config
            config = DQNAgentConfig(
                state_size=state_size,
                action_size=action_size
            )
            self.agent = DQNAgent(config)
        else:
            self.agent = agent
            
        self.visualizer = TradingVisualizer()
        self.history: Dict[str, List[float]] = {"episode_rewards": [], "portfolio_values": []}
        
        # Create results directory if it doesn't exist
        Path("results").mkdir(exist_ok=True)
        
        # Configure file handler for logging
        file_handler = logging.FileHandler("results/training.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

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
        logger.info(f"Starting training for {episodes} episodes...")
        logger.info(f"Initial epsilon: {self.agent.epsilon:.4f}")
        logger.info(f"Initial portfolio value: ${self.env.state.balance:.2f}")

        for episode in range(episodes):
            # Initialize episode
            state = self.env.reset()[0]
            episode_reward = 0.0
            done = False
            steps = 0

            # Run episode
            while not done:
                # Execute step
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store and train
                self.agent.remember(Experience(state, action, reward, next_state, done))
                if len(self.agent.memory) > batch_size:
                    loss = self.agent.replay(batch_size)
                    if steps % 10 == 0:  # Log loss every 10 steps
                        logger.debug(f"Episode {episode}, Step {steps}: Loss = {loss:.4f}")

                # Update state
                state = next_state
                episode_reward += reward
                steps += 1

            # Update target model
            if episode % 10 == 0:
                self.agent.update_target_model()
                logger.info(f"Updated target model at episode {episode}")

            # Update metrics
            self.history["episode_rewards"].append(episode_reward)
            portfolio_value = self.env.state.balance + (self.env.state.shares_held * self.env.df["Close"].iloc[-1])
            self.history["portfolio_values"].append(portfolio_value)

            # Log progress
            if episode % render_interval == 0:
                avg_reward = np.mean(self.history["episode_rewards"][-render_interval:])
                logger.info(f"Episode: {episode}/{episodes}")
                logger.info(f"Average Reward (last {render_interval} episodes): {avg_reward:.2f}")
                logger.info(f"Portfolio Value: ${self.history['portfolio_values'][-1]:.2f}")
                logger.info(f"Epsilon: {self.agent.epsilon:.4f}")
                logger.info(f"Steps in episode: {steps}")
                logger.info("-" * 50)

                # Update visualizations
                self.visualizer.plot_training_history(
                    self.history,
                    save_path="results/training_progress.html"
                )

        logger.info("Training completed!")
        logger.info(f"Final epsilon: {self.agent.epsilon:.4f}")
        logger.info(f"Final portfolio value: ${self.history['portfolio_values'][-1]:.2f}")
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
