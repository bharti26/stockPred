# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data.stock_data import StockData
from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent, DQNAgentConfig
from src.utils.visualization import TradingVisualizer

# %%
# Initialize data handler for test period
ticker = "AAPL"  # Example ticker
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Last year of data

data_handler = StockData(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
data_handler.fetch_data()
data_handler.add_technical_indicators()
data_handler.preprocess_data()

# Print all available columns
print("All available columns:")
print(data_handler.data.columns.tolist())

# Print normalized columns
print("\nNormalized columns:")
normalized_cols = [col for col in data_handler.data.columns if col.endswith('_norm')]
print(normalized_cols)

# Ensure we have all required features
required_features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Lower'
]

# Add any missing features
for feature in required_features:
    if f"{feature}_norm" not in data_handler.data.columns:
        print(f"\nAdding missing feature: {feature}_norm")
        data_handler.data[f"{feature}_norm"] = (data_handler.data[feature] - data_handler.data[feature].mean()) / data_handler.data[feature].std()

# Verify we have all normalized features
normalized_features = [col for col in data_handler.data.columns if col.endswith('_norm')]
print("\nFinal normalized features:")
print(normalized_features)
print(f"\nNumber of normalized features: {len(normalized_features)}")

# Display the data
print("\nFirst few rows of data:")
data_handler.data.head()

# %%
# Initialize environment
env = StockTradingEnv(data_handler.data)

# Create agent config with correct state size
config = DQNAgentConfig(
    state_size=16,  # Fixed state size to match trained model
    action_size=env.action_space.n
)

# Initialize agent
agent = DQNAgent(config)

# Load trained weights
model_path = os.path.join(os.getcwd(), "models/dqn_AAPL.pth")
agent.load(model_path)

print(f"\nModel loaded from {model_path}")

# %%
def evaluate_model(env, agent, initial_balance=10000.0):
    """Evaluate model performance on test data."""
    state = env.reset()[0]
    done = False
    
    # Track trading history
    history = {
        'dates': [],
        'actions': [],
        'prices': [],
        'shares': [],
        'balance': [],
        'portfolio_value': []
    }
    
    while not done:
        # Get model's action
        action = agent.act(state, training=False)  # Use exploitation only
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Record trading history
        current_date = env.df.index[env.state.current_step]
        current_price = env.df.iloc[env.state.current_step]["Close"]
        
        history['dates'].append(current_date)
        history['actions'].append(action)
        history['prices'].append(current_price)
        history['shares'].append(env.state.shares_held)
        history['balance'].append(env.state.balance)
        history['portfolio_value'].append(env.state.balance + (env.state.shares_held * current_price))
        
        state = next_state
    
    return pd.DataFrame(history)

# %%
# Run evaluation
trading_history = evaluate_model(env, agent)

# Display trading history
trading_history.head()

# %%
def analyze_performance(trading_history):
    """Analyze trading performance and calculate metrics."""
    # Calculate returns
    initial_value = trading_history['portfolio_value'].iloc[0]
    final_value = trading_history['portfolio_value'].iloc[-1]
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    # Calculate buy and hold return for comparison
    initial_price = trading_history['prices'].iloc[0]
    final_price = trading_history['prices'].iloc[-1]
    buy_hold_return = ((final_price - initial_price) / initial_price) * 100
    
    # Count trades
    buy_trades = (trading_history['actions'] == 1).sum()
    sell_trades = (trading_history['actions'] == 2).sum()
    
    # Calculate profit/loss per trade
    trades = trading_history[trading_history['actions'] != 0]  # Filter out hold actions
    trade_returns = []
    
    for i in range(len(trades) - 1):
        if trades['actions'].iloc[i] == 1:  # Buy
            buy_price = trades['prices'].iloc[i]
            sell_price = trades['prices'].iloc[i + 1]
            trade_return = ((sell_price - buy_price) / buy_price) * 100
            trade_returns.append(trade_return)
    
    avg_trade_return = np.mean(trade_returns) if trade_returns else 0
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'avg_trade_return': avg_trade_return,
        'total_trades': buy_trades + sell_trades
    }

# %%
# Analyze performance
performance = analyze_performance(trading_history)

print(f"Total Return: {performance['total_return']:.2f}%")
print(f"Buy & Hold Return: {performance['buy_hold_return']:.2f}%")
print(f"Number of Buy Trades: {performance['buy_trades']}")
print(f"Number of Sell Trades: {performance['sell_trades']}")
print(f"Average Trade Return: {performance['avg_trade_return']:.2f}%")
print(f"Total Number of Trades: {performance['total_trades']}")

# %%
def plot_trading_performance(trading_history):
    """Plot trading performance metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio value and stock price
    plt.subplot(2, 1, 1)
    plt.plot(trading_history['dates'], trading_history['portfolio_value'], label='Portfolio Value')
    plt.plot(trading_history['dates'], trading_history['prices'] * 100, label='Stock Price (scaled)')
    plt.title('Portfolio Value vs Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    
    # Plot trading actions
    plt.subplot(2, 1, 2)
    buy_points = trading_history[trading_history['actions'] == 1]
    sell_points = trading_history[trading_history['actions'] == 2]
    
    plt.plot(trading_history['dates'], trading_history['prices'], label='Stock Price')
    plt.scatter(buy_points['dates'], buy_points['prices'], color='green', label='Buy', marker='^')
    plt.scatter(sell_points['dates'], sell_points['prices'], color='red', label='Sell', marker='v')
    plt.title('Trading Actions')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
# Plot trading performance
plot_trading_performance(trading_history)

# %%
def calculate_dollar_returns(trading_history):
    """Calculate dollar profits and losses."""
    initial_balance = trading_history['portfolio_value'].iloc[0]
    final_balance = trading_history['portfolio_value'].iloc[-1]
    
    # Calculate total profit/loss
    total_profit_loss = final_balance - initial_balance
    
    # Calculate profit/loss per trade
    trades = trading_history[trading_history['actions'] != 0]
    trade_profits = []
    
    for i in range(len(trades) - 1):
        if trades['actions'].iloc[i] == 1:  # Buy
            shares = trades['shares'].iloc[i]
            buy_price = trades['prices'].iloc[i]
            sell_price = trades['prices'].iloc[i + 1]
            trade_profit = shares * (sell_price - buy_price)
            trade_profits.append(trade_profit)
    
    avg_trade_profit = np.mean(trade_profits) if trade_profits else 0
    
    return {
        'total_profit_loss': total_profit_loss,
        'avg_trade_profit': avg_trade_profit,
        'winning_trades': len([p for p in trade_profits if p > 0]),
        'losing_trades': len([p for p in trade_profits if p < 0])
    }

# %%
# Calculate dollar returns
dollar_returns = calculate_dollar_returns(trading_history)

print(f"Total Profit/Loss: ${dollar_returns['total_profit_loss']:.2f}")
print(f"Average Profit per Trade: ${dollar_returns['avg_trade_profit']:.2f}")
print(f"Number of Winning Trades: {dollar_returns['winning_trades']}")
print(f"Number of Losing Trades: {dollar_returns['losing_trades']}") 