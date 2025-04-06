# System Architecture

## Overview

The Stock Trading AI system is built with a modular architecture that separates concerns into distinct components while maintaining efficient communication between them. This document details the system's architecture and the interaction between its components.

## Core Components

### 1. DQN Agent (src/models/dqn_agent.py)
The neural network-based agent that makes trading decisions:
- **Architecture**: 
  - 3-layer neural network
  - Input layer: Dynamic state size
  - Hidden layers: 64 units with ReLU activation
  - Output layer: 3 units (buy, sell, hold)
- **Input Processing**: 
  - Flexible state handling (numpy arrays, lists, tuples)
  - Automatic shape correction and type conversion
  - Support for tuple states from environment
- **Decision Making**: 
  - Q-value computation for actions
  - Epsilon-greedy exploration with decay
  - Automatic model selection based on ticker
- **Learning**: 
  - Experience replay with prioritized sampling
  - Target network updates for stability
  - Robust error handling during training

### 2. Trading Environment (src/env/trading_env.py)
The simulation environment that interfaces with the agent:
- **State Management**: 
  - Enhanced market condition tracking
  - Improved portfolio state representation
  - Tuple state format for additional info
- **Action Processing**: 
  - Validated trading decisions
  - Transaction cost consideration
  - Position size optimization
- **Reward Calculation**: 
  - Risk-adjusted performance metrics
  - Multi-factor reward computation
  - Transaction cost penalties
- **Constraint Handling**: 
  - Dynamic trading limits
  - Risk management rules
  - Market impact consideration

### 3. Data Processing (src/data/)
Handles all data-related operations:
- **Market Data**: 
  - Multi-interval support (1d, 1h, 5m)
  - Automatic interval selection
  - Fallback mechanisms for missing data
- **Technical Analysis**: 
  - Enhanced indicator calculation
  - Real-time updates with error handling
  - Customizable indicator parameters
- **Data Validation**: 
  - Comprehensive data quality checks
  - Missing data interpolation
  - Outlier detection and handling
- **Caching**: 
  - Efficient data storage with versioning
  - Automatic cache invalidation
  - Memory-optimized storage

### 4. Visualization System (src/utils/)
Provides real-time visual feedback:
- **Interactive Charts**: 
  - Dynamic price and indicator plots
  - Real-time trade annotations
  - Custom color schemes
- **Performance Metrics**: 
  - Enhanced trading statistics
  - Risk-adjusted returns
  - Portfolio analytics
- **Real-time Updates**: 
  - Efficient data streaming
  - Error state visualization
  - Status indicators
- **Export Capabilities**: 
  - Multiple format support
  - Custom report generation
  - Data export options

## Component Interactions

### Data Flow
1. Market data flows from Yahoo Finance to Data Processing with error handling
2. Processed data feeds into Trading Environment with validation
3. Environment state is passed to DQN Agent with type safety
4. Agent's actions return to Environment with verification
5. Results flow to Visualization System with status updates

### Communication Protocols
- **Data Processing → Environment**: Validated DataFrame updates
- **Environment → Agent**: Typed state vectors with metadata
- **Agent → Environment**: Validated action integers
- **Environment → Visualization**: Error-aware event updates

## System Requirements

### Hardware
- CPU: Multi-core processor
- RAM: 8GB minimum (16GB recommended)
- Storage: 1GB for model checkpoints and data cache

### Software
- Python 3.8+
- PyTorch
- Pandas
- Plotly
- Additional packages in requirements.txt

### Type Safety and Code Quality
- Comprehensive static type checking
- Enhanced error handling and logging
- Automated testing with pytest
- Performance benchmarking
- Code coverage tracking

## Performance Considerations

### Optimization
- Improved batch processing
- Enhanced memory management
- Configurable update intervals
- Automatic resource scaling

### Scalability
- Model versioning support
- Multi-stock training
- Parallel data processing
- Distributed training support

## Security

### Data Protection
- Secure API key management
- Enhanced error logging
- Rate limiting and quotas
- Data validation layers

### Error Handling
- Comprehensive error states
- Automatic recovery mechanisms
- Detailed error reporting
- User-friendly error messages

## Future Extensibility

### Planned Features
- Advanced model architectures
- Custom indicator builder
- Enhanced risk management
- Live trading integration

### Integration Points
- Multiple data source support
- Custom model integration
- External API connections
- Monitoring system hooks 