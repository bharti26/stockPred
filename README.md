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
- Automatic model selection based on stock ticker
- Robust error handling and data validation
- Flexible data fetching with multiple time intervals

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
make benchmark # Run performance benchmarks
```

## Running the Code

### Training Models

The system supports training on multiple stocks simultaneously:

```bash
# Train on predefined stocks
python src/train_models.py

# Train on top 10 stocks (default)
make train

# Or with custom parameters
python src/main.py --start-date 2020-01-01 --end-date 2023-12-31 --episodes 100
```

Training will:
- Train models for predefined stock tickers
- Train separate optimized and default models
- Save models to the `models` directory
- Display training progress and metrics
- Show a final summary of results

### Running Tests

```bash
make test
```

The test suite includes:
- Unit tests for all components
- Performance benchmarks
- Integration tests
- Real-time data handling tests
- Model architecture tests

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
- Interactive stock selection with automatic model loading
- Technical indicators display
- Model prediction visualization
- Trade history and performance metrics
- Portfolio analysis
- Top 10 stocks tracking with:
  - Current prices
  - Price changes
  - Trading volume
  - Color-coded performance indicators
- Robust error handling and status display
- Automatic data refresh and model updates

## Project Structure

```
stockPred/
├── src/
│   ├── data/
│   │   ├── stock_data.py      # Stock data fetching and preprocessing
│   │   ├── realtime_data.py   # Real-time data streaming with error handling
│   │   └── top_stocks.py      # Top stocks tracking
│   ├── models/
│   │   └── dqn_agent.py       # DQN agent with improved state handling
│   ├── env/
│   │   └── trading_env.py     # Trading environment with enhanced rewards
│   ├── dashboard/
│   │   ├── app.py            # Main dashboard with error handling
│   │   └── run.py            # Dashboard runner with port configuration
│   └── main.py               # Main training script
├── tests/                    # Comprehensive test suite
├── models/                   # Trained model files (optimized and default)
├── results/                  # Training results and plots
├── requirements.txt          # Project dependencies
├── Makefile                 # Make commands for common tasks
└── README.md                # This file
```

## Model Architecture

The system uses a Deep Q-Network (DQN) with:
- 3-layer neural network architecture:
  - Input layer: State size (varies by stock)
  - Hidden layers: 64 units each with ReLU activation
  - Output layer: 3 units (buy, sell, hold)
- Experience replay for stable learning
- Target network for stable Q-value estimation
- Epsilon-greedy exploration strategy
- Huber loss for robust training
- Improved state handling for various input formats
- Automatic model selection based on stock characteristics

### Data Handling Improvements
- Multiple time interval support (1d, 1h, 5m)
- Robust error handling for data fetching
- Automatic retry mechanism for data streams
- Graceful fallback for missing data
- Enhanced technical indicators

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 