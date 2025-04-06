# Class Diagrams

## Core Components

### DQN Agent
```
┌───────────────────────┐
│      DQNAgent        │
├───────────────────────┤
│ - policy_net         │
│ - target_net         │
│ - memory_buffer      │
│ - epsilon            │
├───────────────────────┤
│ + act()              │
│ + remember()         │
│ + replay()           │
│ + update_target()    │
└───────────────────────┘
         ▲
         │
┌───────────────────────┐
│     QNetwork         │
├───────────────────────┤
│ - layers             │
│ - optimizer          │
├───────────────────────┤
│ + forward()          │
│ + save()             │
│ + load()             │
└───────────────────────┘
```

### Trading Environment
```
┌───────────────────────┐
│   TradingEnv         │
├───────────────────────┤
│ - data               │
│ - position           │
│ - balance            │
│ - done               │
├───────────────────────┤
│ + reset()            │
│ + step()             │
│ + render()           │
│ + close()            │
└───────────────────────┘
         ▲
         │
┌───────────────────────┐
│   PositionManager    │
├───────────────────────┤
│ - current_position   │
│ - entry_price        │
├───────────────────────┤
│ + open_position()    │
│ + close_position()   │
│ + update_pnl()       │
└───────────────────────┘
```

### Data Processing
```
┌───────────────────────┐
│    RealTimeData      │
├───────────────────────┤
│ - ticker             │
│ - interval           │
│ - buffer             │
├───────────────────────┤
│ + start_streaming()  │
│ + stop_streaming()   │
│ + get_latest_data()  │
└───────────────────────┘
         ▲
         │
┌───────────────────────┐
│  TechnicalAnalysis   │
├───────────────────────┤
│ - data               │
│ - indicators         │
├───────────────────────┤
│ + calculate_all()    │
│ + update_indicators()│
└───────────────────────┘
```

### Visualization
```
┌───────────────────────┐
│ RealtimeVisualizer   │
├───────────────────────┤
│ - fig                │
│ - layout_settings    │
│ - colors             │
├───────────────────────┤
│ + initialize_plot()  │
│ + update_plot()      │
│ + add_trade()        │
└───────────────────────┘
         ▲
         │
┌───────────────────────┐
│ PerformanceMetrics   │
├───────────────────────┤
│ - trades             │
│ - metrics            │
├───────────────────────┤
│ + calculate_metrics()│
│ + update_display()   │
└───────────────────────┘
```

## Class Relationships

### Training System
```
┌─────────────┐     ┌─────────────┐
│  Trainer    │────►│  DQNAgent   │
└─────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ TradingEnv  │◄────┤  QNetwork   │
└─────────────┘     └─────────────┘
```

### Data System
```
┌─────────────┐     ┌─────────────┐
│ RealTimeData│────►│ TechAnalysis│
└─────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ DataBuffer  │◄────┤ Indicators  │
└─────────────┘     └─────────────┘
```

### Visualization System
```
┌─────────────┐     ┌─────────────┐
│ Visualizer  │────►│ Metrics     │
└─────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ Plots       │◄────┤ Analytics   │
└─────────────┘     └─────────────┘
```

## Class Methods

### DQNAgent Methods
```python
class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory: ReplayBuffer = ReplayBuffer(100000)
        self.gamma: float = 0.95
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.995
        self.learning_rate: float = 0.001
        self.policy_net: QNetwork = QNetwork(state_size, action_size)
        self.target_net: QNetwork = QNetwork(state_size, action_size)
    
    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer"""
    
    def replay(self, batch_size: int) -> float:
        """Train on random batch from replay buffer"""
    
    def update_target(self) -> None:
        """Update target network weights"""
```

### TradingEnv Methods
```python
class TradingEnv:
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data
        self.initial_balance = initial_balance
        self.position: PositionManager = PositionManager()
        self.current_step: int = 0
    
    def reset(self, *, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next state"""
    
    def get_state(self) -> np.ndarray:
        """Get current market state"""
    
    def calculate_reward(self) -> float:
        """Calculate reward for current step"""
```

### StockData Methods
```python
class StockData:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data: Optional[pd.DataFrame] = None
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance"""
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """Add technical indicators to the stock data"""
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess data by handling missing values and normalizing features"""
```

### RealTimeData Methods
```python
class RealTimeStockData:
    def __init__(self, ticker: str, interval: str = "1m", buffer_size: int = 100):
        self.ticker = ticker
        self.interval = interval
        self.buffer_size = buffer_size
        self.data = pd.DataFrame()
        self.callbacks: List[Callable[[pd.DataFrame], None]] = []
        self.is_streaming: bool = False
        self.stream_thread: Optional[threading.Thread] = None
    
    def add_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Add a callback function to be called on data updates"""
    
    def start_streaming(self) -> None:
        """Start the real-time data streaming"""
    
    def stop_streaming(self) -> None:
        """Stop the real-time data streaming"""
    
    def get_latest_indicators(self) -> Dict[str, Union[float, None]]:
        """Get the latest technical indicators"""
```

### Visualization Methods
```python
class RealtimeVisualizer:
    def __init__(self, update_interval: float = 1.0):
        self.fig = None
        self.layout_settings: Dict[str, Any] = {}
        self.colors: Dict[str, str] = {}
        self.update_interval = update_interval
        self.update_thread: Optional[threading.Thread] = None
    
    def initialize_plot(self, data: pd.DataFrame) -> None:
        """Set up initial plot structure"""
    
    def update_plot(self, data: pd.DataFrame) -> None:
        """Update plot with new data"""
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add trade marker to plot"""
    
    def start_updating(self, data_callback: Callable[[], pd.DataFrame]) -> None:
        """Start automatic plot updates"""
    
    def stop_updating(self) -> None:
        """Stop automatic plot updates"""
    
    def save_html(self, filename: str) -> None:
        """Save interactive plot as HTML"""
``` 