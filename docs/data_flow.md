# Data Flow Documentation

## Market Data Flow

```
External Source        Processing            Storage              Usage
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Yahoo       │    │  Data        │    │  In-Memory   │    │  Trading     │
│  Finance API │───►│  Validation  │───►│  Buffer      │───►│  Environment │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        │                  │                    │                    │
        │                  ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Interval    │    │  Technical   │    │  Model       │    │  State       │
│  Selection   │    │  Analysis    │    │  Selection   │    │  Vector      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Data Processing Pipeline

### 1. Data Acquisition
- **Real-time Data**
  ```
  Market Feed → Interval Selection → Validation → Normalization → Buffer
  ```
- **Historical Data**
  ```
  API Request → Interval Testing → Data Cleaning → Feature Engineering → Storage
  ```

### 2. Technical Analysis
- **Indicator Calculation**
  ```
  Raw Data → Moving Averages → Oscillators → Volatility → Combined Indicators
  ```
- **Signal Generation**
  ```
  Indicators → Thresholds → Crossovers → Signal Matrix → Error Checking
  ```

### 3. State Preparation
- **Feature Vector**
  ```
  [Price Data | Technical Indicators | Position Info | Market Context | Error States]
  ```
- **Normalization**
  ```
  Raw Values → Type Conversion → Scaling → Standardization → Normalized Vector
  ```

## Data Types and Formats

### Market Data
```python
{
    'timestamp': datetime,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    'adjusted_close': float,
    'interval': str,  # '1d', '1h', or '5m'
    'data_quality': float  # Quality score
}
```

### Technical Indicators
```python
{
    'sma_20': float,
    'sma_50': float,
    'rsi_14': float,
    'macd': float,
    'macd_signal': float,
    'bb_upper': float,
    'bb_lower': float,
    'atr': float,
    'obv': float,
    'roc': float,
    'calculation_status': str,  # 'success', 'partial', 'failed'
    'error_info': Optional[str]  # Error details if any
}
```

### State Vector
```python
[
    # Price Features (normalized)
    price_change,        # [-1, 1]
    volume_change,       # [-1, 1]
    price_volatility,    # [0, 1]
    
    # Technical Indicators (normalized)
    rsi,                # [0, 1]
    macd_histogram,     # [-1, 1]
    bb_position,        # [-1, 1]
    
    # Position Information
    position_size,      # [0, 1]
    unrealized_pnl,     # [-1, 1]
    time_in_position,   # [0, 1]
    
    # Additional Context
    data_quality,       # [0, 1]
    error_state,        # [0, 1]
    interval_type       # One-hot encoded
]
```

## Data Update Frequency

### Real-time Updates
- Market Data: Adaptive intervals (1d, 1h, 5m)
- Technical Indicators: On each price update with validation
- State Vector: On each environment step with error checking

### Batch Processing
- Historical Data: Daily updates with interval optimization
- Model Training: Every 1000 steps with error tracking
- Performance Metrics: Every trading day with quality scores

## Data Validation Rules

### Price Data
- Non-negative values
- Consistent OHLC relationships
- Maximum percentage changes
- Volume thresholds
- Interval consistency

### Technical Indicators
- Value ranges
- Calculation periods
- Missing data handling
- Outlier detection
- Calculation status tracking

### State Vector
- Normalization bounds
- Feature completeness
- Temporal consistency
- Correlation checks
- Type validation

## Error Handling

### Data Quality Issues
```python
try:
    # Multi-level validation
    validate_price_data(data)
    validate_indicators(indicators)
    validate_state_vector(state)
    validate_interval_consistency(interval)
except DataValidationError as e:
    # Enhanced recovery procedures
    if e.error_type == 'interval':
        try_different_interval()
    elif e.error_type == 'data':
        use_fallback_data()
    elif e.error_type == 'calculation':
        use_simplified_indicators()
    
    notify_system_monitor(e)
    log_validation_failure(e)
    update_error_state(e)
```

### Missing Data
```python
def handle_missing_data(data, interval):
    if missing_values > threshold:
        if interval == '5m':
            return try_hourly_data()
        elif interval == '1h':
            return try_daily_data()
        else:
            return interpolate_values(data)
    else:
        return forward_fill(data)
```

## Monitoring and Logging

### Data Quality Metrics
- Data completeness by interval
- Update frequency success rate
- Processing latency distribution
- Error rates by type
- Interval switching frequency

### System Health
- Buffer utilization by interval
- Processing queue length
- Computation time distribution
- Memory usage patterns
- Model loading status

### Audit Trail
- Data source and interval
- Processing steps with timing
- Validation results with details
- Error records with context
- Recovery actions taken 