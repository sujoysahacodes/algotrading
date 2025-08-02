# Algorithmic Trading System

A comprehensive algorithmic trading application implementing trend-following and mean-reversion strategies with real-time execution capabilities.

## Project Structure

```
# 📈 Algorithmic Trading System

A comprehensive algorithmic trading system implementing trend-following and mean-reversion strategies with real-time data analysis, backtesting, and risk management.

## 🎯 Overview

This project implements a complete algorithmic trading platform featuring:

- **Multiple Trading Strategies**: EMA crossover and Z-score mean reversion
- **Real-time Data**: Yahoo Finance integration with live market data
- **Interactive Dashboard**: Streamlit-based web interface
- **Comprehensive Backtesting**: Performance analysis with risk metrics
- **Jupyter Analysis**: Interactive notebooks for strategy research
- **Production Ready**: Docker containerization and database support

## 🚀 Quick Start

### Option 1: Streamlit Dashboard (Recommended)
```bash
# Clone repository
git clone https://github.com/sujoysahacodes/algotrading.git
cd algotrading

# Install dependencies
pip install -r requirements-minimal.txt

# Launch dashboard
streamlit run streamlit_dashboard.py
```

**Access at**: http://localhost:8501

### Option 2: Jupyter Analysis
```bash
# Install dependencies
pip install -r requirements-minimal.txt

# Start Jupyter
jupyter notebook

# Open: notebooks/Strategy_Analysis.ipynb
```

### Option 3: Python Scripts
```bash
# Test components
python test_notebook.py

# Run simplified trading
python simple_trading.py
```

## 📊 Features

### **Trading Strategies**
- ✅ **EMA Crossover**: Trend-following with 12/26 EMAs
- ✅ **Z-Score Mean Reversion**: Statistical reversal strategy
- ✅ **Configurable Parameters**: Customizable strategy settings
- ✅ **Signal Generation**: Buy/sell signal detection

### **Data & Analysis**
- ✅ **Real Market Data**: Yahoo Finance API integration
- ✅ **Multiple Symbols**: AAPL, MSFT, GOOGL, SPY, TSLA, etc.
- ✅ **Technical Indicators**: EMA, SMA, Z-Score, Bollinger Bands
- ✅ **Performance Metrics**: Returns, Sharpe ratio, drawdown analysis

### **Interactive Dashboard**
- ✅ **Real-time Configuration**: Symbol and parameter selection
- ✅ **Live Charts**: Price movements with trading signals
- ✅ **Performance Analysis**: Strategy vs buy-and-hold comparison
- ✅ **Risk Metrics**: Volatility, drawdown, win rate analysis

### **Development Tools**
- ✅ **Jupyter Notebooks**: Interactive strategy research
- ✅ **Comprehensive Tests**: Validation and error checking
- ✅ **Docker Support**: Containerized deployment
- ✅ **Database Integration**: TimescaleDB for time-series data

## 🏗️ Project Structure

```
algotrading/
├── 📊 streamlit_dashboard.py      # Main Streamlit dashboard
├── 🧪 simple_trading.py          # Simplified trading modules
├── 📓 notebooks/                 # Jupyter analysis notebooks
│   └── Strategy_Analysis.ipynb   # Comprehensive strategy analysis
├── 🔧 src/                       # Core trading modules
│   ├── strategies/               # Trading strategy implementations
│   ├── data/                     # Data providers and management
│   ├── backtesting/              # Backtesting engine
│   ├── risk/                     # Risk management
│   └── dashboard/                # Advanced dashboard components
├── 🐳 docker/                    # Docker configuration
├── 📋 tests/                     # Test suite
├── 📜 scripts/                   # Utility scripts
└── 📖 docs/                      # Documentation
```

## 💡 Usage Examples

### Dashboard Analysis
1. **Select Symbol**: Choose from AAPL, MSFT, GOOGL, etc.
2. **Configure Strategy**: EMA crossover or Z-score parameters
3. **Set Date Range**: Historical analysis period
4. **Run Analysis**: View results across multiple tabs
5. **Compare Performance**: Strategy vs benchmark returns

### Jupyter Research
```python
from simple_trading import data_manager, SimpleTradingStrategy
from datetime import datetime

# Fetch data
data = data_manager.get_historical_data('AAPL', datetime(2023,1,1), datetime(2024,1,1))

# Create strategy
class MyStrategy(SimpleTradingStrategy):
    def calculate_indicators(self, data):
        indicators = data.copy()
        indicators['ema_12'] = self.calculate_ema(data['close'], 12)
        indicators['ema_26'] = self.calculate_ema(data['close'], 26)
        return indicators

# Analyze results
strategy = MyStrategy()
results = strategy.calculate_indicators(data)
```

## 📈 Strategy Performance

### Sample Results (AAPL 2023)
- **EMA Crossover**: 15.14% return, 0.75 Sharpe ratio
- **Z-Score Reversion**: Statistical mean reversion signals
- **Risk Metrics**: 20.11% volatility, controlled drawdown

## 🛠️ Installation

### Requirements
- Python 3.8+
- pandas, numpy, scipy
- yfinance (market data)
- streamlit (dashboard)
- matplotlib, plotly (visualization)
- jupyter (analysis)

### Full Installation
```bash
# Clone repository
git clone https://github.com/sujoysahacodes/algotrading.git
cd algotrading

# Install minimal dependencies
pip install -r requirements-minimal.txt

# Or install full package
pip install -e .
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d
```

## 🧪 Testing

```bash
# Test all components
python test_notebook.py

# Test specific modules
python test_imports.py
python simple_trading.py
```

## 📊 Screenshots

### Streamlit Dashboard
- Interactive parameter configuration
- Real-time price charts with signals
- Performance metrics and comparisons
- Technical indicator analysis

### Jupyter Analysis
- Mathematical strategy foundations
- Comprehensive backtesting results
- Risk analysis and visualization
- Strategy comparison tables

## 🚀 Advanced Features

### Production Deployment
- **Docker**: Containerized application
- **TimescaleDB**: Time-series database
- **Redis**: Caching layer
- **Monitoring**: Performance tracking

### Risk Management
- **VaR Calculation**: Value at Risk metrics
- **Position Sizing**: Kelly Criterion implementation
- **Drawdown Monitoring**: Risk control measures
- **Portfolio Limits**: Exposure management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-strategy`)
3. Commit changes (`git commit -am 'Add new strategy'`)
4. Push to branch (`git push origin feature/new-strategy`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` folder
- **Examples**: Jupyter notebooks in `/notebooks`

## 🏆 Acknowledgments

- Yahoo Finance for market data
- Streamlit for dashboard framework
- Plotly for interactive visualizations
- Scientific Python ecosystem (pandas, numpy, scipy)

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Not financial advice. Trading involves risk of loss.
├── src/
│   ├── strategies/           # Trading strategies implementation
│   ├── data/                # Data fetching and processing
│   ├── brokers/             # Broker API integrations
│   ├── backtesting/         # Backtesting engine
│   ├── risk/                # Risk management
│   ├── utils/               # Utility functions
│   └── dashboard/           # Streamlit dashboard
├── config/                  # Configuration files
├── tests/                   # Unit tests
├── docker/                  # Docker configurations
├── notebooks/               # Jupyter notebooks for analysis
└── scripts/                 # Deployment and maintenance scripts
```

## Features

### Part I: Core Strategies
- **Trend-Following Strategies**:
  - EMA Crossover with ADX confirmation
  - MACD with trend strength filtering
  - Custom FX trend indicator (price ratio method)
  
- **Mean-Reversion Strategies**:
  - Z-score based signals
  - Ornstein-Uhlenbeck process modeling
  - Statistical arbitrage

### Part II: Broker Integration
- **Supported Brokers**:
  - Alpaca Markets (REST API + WebSocket)
  - Interactive Brokers (TWS API)
  - Paper trading for testing

- **Data Sources**:
  - OpenBB for professional data
  - High-frequency data (15min and below)
  - Real-time market data with quality checks

### Part III: Risk Management & Production
- **Risk Controls**:
  - Position sizing and portfolio limits
  - Real-time VaR monitoring
  - Drawdown protection

- **Production Features**:
  - Docker containerization
  - Database integration (TimescaleDB)
  - Automated scheduling with cron
  - Real-time dashboard with Streamlit

## Quick Start

1. **Installation**:
```bash
pip install -r requirements.txt
```

2. **Configuration**:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your API keys and preferences
```

3. **Run Backtesting**:
```bash
python scripts/run_backtest.py --strategy ema_adx --symbol AAPL --start 2023-01-01 --end 2024-01-01
```

4. **Start Dashboard**:
```bash
streamlit run src/dashboard/app.py
```

5. **Live Trading** (Paper Mode):
```bash
python scripts/run_live.py --mode paper --strategy mean_reversion
```

## Strategy Details

### Trend-Following: EMA-ADX Strategy
- **EMA Periods**: 12, 26 (fast/slow)
- **ADX Threshold**: 25 (trend strength filter)
- **Signal**: EMA crossover confirmed by ADX > threshold

### Mean-Reversion: Z-Score Strategy
- **Lookback Window**: 20 periods
- **Entry Threshold**: |Z-score| > 2.0
- **Exit Threshold**: |Z-score| < 0.5

## Risk Management
- **Position Sizing**: Kelly Criterion with 0.25 fraction
- **Stop Loss**: 2% of position value
- **Portfolio Heat**: Maximum 10% risk per trade
- **VaR Limit**: 5% daily portfolio VaR

## Testing
```bash
pytest tests/ -v
```

## Docker Deployment
```bash
docker-compose up -d
```

## License
MIT License - See LICENSE file for details

## Disclaimer
This software is for educational and research purposes only. Do not use for live trading without proper testing and risk management.