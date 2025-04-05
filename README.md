# StockPred - Stock Prediction with Reinforcement Learning

A tool for predicting stock movements using reinforcement learning techniques with interactive visualizations and a web dashboard.

## Features

- Multi-stock training and prediction
- Real-time stock data visualization
- Interactive stock selection
- Technical indicators display
- Model prediction visualization
- Trade history and performance metrics
- Portfolio analysis
- Top 10 stocks tracking

## Setup

### Prerequisites

- Python 3.8+
- pip
- make
- virtualenv (optional, but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stockPred.git
   cd stockPred
   ```

2. Set up the development environment using make:
   ```bash
   # This will create a virtual environment and install all dependencies
   make setup
   ```

   Alternatively, you can install the package directly:
   ```bash
   make install
   ```

3. Create a `.env` file in the project root:
   ```
   # Sets the PYTHONPATH to the project root directory
   PYTHONPATH=/path/to/your/stockPred
   ```

### Available Make Commands

The project includes several make commands for common tasks:

```bash
make setup     # Set up development environment
make install   # Install package
make test      # Run tests
make lint      # Run linters
make format    # Format code
make clean     # Clean up build files
make train     # Train models
make dashboard # Start dashboard
```

## Running the Code

### Training Models

The system supports training on multiple stocks simultaneously:

```bash
# Train on top 10 stocks (default)
make train

# Or with custom parameters
python src/main.py --start-date 2020-01-01 --end-date 2023-12-31 --episodes 100
```

Training will:
- Fetch the top 10 most active stocks
- Train a separate model for each stock
- Save models to the `models` directory
- Display training progress and metrics
- Show a final summary of results

### Running Tests

```bash
make test
```

### Running the Dashboard

```bash
# Start the dashboard
make dashboard

# Or with custom parameters
python src/dashboard/run.py --port 8050 --debug --auto-open
```

The dashboard will be available at `http://localhost:8050` (or your specified port).

## Dashboard Features

- Real-time stock data visualization
- Interactive stock selection
- Technical indicators display
- Model prediction visualization
- Trade history and performance metrics
- Portfolio analysis
- Top 10 stocks tracking with:
  - Current prices
  - Price changes
  - Trading volume
  - Color-coded performance indicators

## Project Structure

```
stockPred/
├── src/
│   ├── data/
│   │   ├── stock_data.py      # Stock data fetching and preprocessing
│   │   ├── realtime_data.py   # Real-time data streaming
│   │   └── top_stocks.py      # Top stocks tracking
│   ├── models/
│   │   └── dqn_agent.py       # DQN agent implementation
│   ├── env/
│   │   └── trading_env.py     # Trading environment
│   ├── dashboard/
│   │   ├── app.py            # Main dashboard application
│   │   └── run.py            # Dashboard runner
│   └── main.py               # Main training script
├── tests/                    # Test files
├── models/                   # Trained model files
├── results/                  # Training results and plots
├── requirements.txt          # Project dependencies
├── Makefile                 # Make commands for common tasks
└── README.md                # This file
```

## Model Architecture

The system uses a Deep Q-Network (DQN) with:
- Experience replay for stable learning
- Target network for stable Q-value estimation
- Epsilon-greedy exploration strategy
- Huber loss for robust training
- Configurable hyperparameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 