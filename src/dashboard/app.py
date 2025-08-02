"""
Algorithmic Trading Dashboard

A Streamlit-based dashboard for monitoring trading strategies,
backtesting results, and risk metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.manager import data_manager
from strategies import EMAAdxStrategy, ZScoreMeanReversionStrategy, FXTrendStrategy
from backtesting import BacktestEngine
from risk import RiskManager
from utils import config

# Page configuration
st.set_page_config(
    page_title="AlgoTrading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive { color: #26C281; }
    .negative { color: #F63366; }
    .neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(symbol, start_date, end_date, frequency):
    """Load market data with caching."""
    try:
        data = data_manager.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_strategies():
    """Create strategy instances."""
    strategies = {}
    
    # EMA-ADX Strategy
    ema_config = config.get('strategies.trend_following.ema_adx', {})
    strategies['EMA-ADX'] = EMAAdxStrategy(ema_config)
    
    # Z-Score Mean Reversion
    zscore_config = config.get('strategies.mean_reversion.zscore', {})
    strategies['Z-Score Reversion'] = ZScoreMeanReversionStrategy(zscore_config)
    
    # FX Trend Strategy
    fx_config = {
        'short_interval': '30s',
        'long_interval': '5min',
        'trend_threshold': 0.02
    }
    strategies['FX Trend'] = FXTrendStrategy(fx_config)
    
    return strategies

def plot_price_and_signals(data, signals, strategy_name):
    """Plot price chart with trading signals."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.03,
        subplot_titles=[f'{strategy_name} - Price & Signals', 'Volume'],
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add signals
    buy_signals = [s for s in signals if s.signal_type.name == 'BUY']
    sell_signals = [s for s in signals if s.signal_type.name == 'SELL']
    
    if buy_signals:
        buy_times = [s.timestamp for s in buy_signals]
        buy_prices = [data.loc[s.timestamp]['close'] for s in buy_signals if s.timestamp in data.index]
        
        fig.add_trace(
            go.Scatter(
                x=buy_times[:len(buy_prices)],
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal'
            ),
            row=1, col=1
        )
    
    if sell_signals:
        sell_times = [s.timestamp for s in sell_signals]
        sell_prices = [data.loc[s.timestamp]['close'] for s in sell_signals if s.timestamp in data.index]
        
        fig.add_trace(
            go.Scatter(
                x=sell_times[:len(sell_prices)],
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal'
            ),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color='rgba(128,128,128,0.5)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def plot_backtest_results(result):
    """Plot backtest results."""
    if not result.portfolio_values:
        st.warning("No backtest results to display")
        return None
    
    # Create portfolio value chart
    portfolio_df = pd.DataFrame({
        'Date': result.timestamps,
        'Portfolio Value': result.portfolio_values,
        'Returns': [0] + result.returns[1:] if len(result.returns) > 1 else [0]
    })
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.03,
        subplot_titles=['Portfolio Value', 'Daily Returns'],
        row_width=[0.7, 0.3]
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_df['Date'],
            y=portfolio_df['Portfolio Value'],
            name='Portfolio Value',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Benchmark line
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        row=1, col=1
    )
    
    # Returns
    colors = ['green' if r > 0 else 'red' for r in portfolio_df['Returns']]
    fig.add_trace(
        go.Bar(
            x=portfolio_df['Date'],
            y=portfolio_df['Returns'],
            name='Daily Returns',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True
    )
    
    return fig

def display_performance_metrics(result):
    """Display performance metrics in a nice format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{result.total_return:.2%}",
            delta=None,
            delta_color="normal"
        )
        st.metric(
            "Volatility",
            f"{result.volatility:.2%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}",
            delta=None,
            delta_color="normal" if result.sharpe_ratio > 1 else "inverse"
        )
        st.metric(
            "Win Rate",
            f"{result.win_rate:.1%}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{result.max_drawdown:.2%}",
            delta=None,
            delta_color="inverse"
        )
        st.metric(
            "Number of Trades",
            f"{result.num_trades}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Annualized Return",
            f"{result.annualized_return:.2%}",
            delta=None,
            delta_color="normal"
        )
        st.metric(
            "Avg Trade Return",
            f"{result.avg_trade_return:.2%}",
            delta=None
        )

def main():
    """Main dashboard function."""
    st.title("🚀 Algorithmic Trading Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Symbol selection
        symbol = st.selectbox(
            "Select Symbol",
            ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "IWM"],
            index=0
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
        
        # Data frequency
        frequency = st.selectbox(
            "Data Frequency",
            ["1d", "1h", "15min", "5min"],
            index=0
        )
        
        # Initial capital
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000
        )
        
        st.markdown("---")
        
        # Strategy selection
        st.header("Strategy Selection")
        available_strategies = create_strategies()
        selected_strategies = st.multiselect(
            "Select Strategies",
            list(available_strategies.keys()),
            default=list(available_strategies.keys())
        )
    
    # Main content
    if not selected_strategies:
        st.warning("Please select at least one strategy from the sidebar.")
        return
    
    # Load data
    with st.spinner(f"Loading data for {symbol}..."):
        data = load_data(symbol, start_date, end_date, frequency)
    
    if data.empty:
        st.error("Failed to load data. Please check your inputs.")
        return
    
    st.success(f"Loaded {len(data)} data points for {symbol}")
    
    # Display raw data
    with st.expander("View Raw Data"):
        st.dataframe(data.tail(10))
    
    # Run backtests
    st.header("Backtesting Results")
    
    # Initialize backtest engine
    backtest_config = {
        'initial_capital': initial_capital,
        'commission': 0.001,
        'slippage': 0.0005
    }
    engine = BacktestEngine(backtest_config)
    
    # Results storage
    results = {}
    
    # Run backtests for selected strategies
    for strategy_name in selected_strategies:
        with st.spinner(f"Running backtest for {strategy_name}..."):
            strategy = available_strategies[strategy_name]
            
            try:
                result = engine.run_backtest(
                    strategy=strategy,
                    data=data,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                results[strategy_name] = result
                
            except Exception as e:
                st.error(f"Error running backtest for {strategy_name}: {e}")
    
    # Display results
    if results:
        # Strategy comparison
        if len(results) > 1:
            st.subheader("Strategy Comparison")
            
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Strategy': name,
                    'Total Return': f"{result.total_return:.2%}",
                    'Annualized Return': f"{result.annualized_return:.2%}",
                    'Volatility': f"{result.volatility:.2%}",
                    'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                    'Max Drawdown': f"{result.max_drawdown:.2%}",
                    'Win Rate': f"{result.win_rate:.1%}",
                    'Trades': result.num_trades
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        # Individual strategy results
        for strategy_name, result in results.items():
            st.subheader(f"{strategy_name} Results")
            
            # Performance metrics
            display_performance_metrics(result)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Portfolio value chart
                portfolio_fig = plot_backtest_results(result)
                if portfolio_fig:
                    st.plotly_chart(portfolio_fig, use_container_width=True)
            
            with col2:
                # Price and signals chart
                strategy_obj = available_strategies[strategy_name]
                signals = strategy_obj.get_signals_history()
                
                if signals:
                    price_fig = plot_price_and_signals(data, signals, strategy_name)
                    st.plotly_chart(price_fig, use_container_width=True)
                else:
                    st.info("No signals generated for this strategy")
            
            # Trade details
            with st.expander(f"View {strategy_name} Trade Details"):
                if result.trades:
                    trades_df = pd.DataFrame(result.trades)
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades executed")
            
            st.markdown("---")
    
    # Risk Analysis
    if results:
        st.header("Risk Analysis")
        
        risk_manager = RiskManager()
        
        for strategy_name, result in results.items():
            if result.returns:
                returns_series = pd.Series(result.returns)
                
                # Generate risk report
                risk_report = risk_manager.generate_risk_report(
                    portfolio_returns=returns_series,
                    positions={},  # Simplified for demo
                    portfolio_value=result.portfolio_values[-1] if result.portfolio_values else initial_capital
                )
                
                st.subheader(f"{strategy_name} Risk Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "VaR (95%)",
                        f"{risk_report.get('var_95_1day', 0):.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Expected Shortfall",
                        f"{risk_report.get('expected_shortfall_95', 0):.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Current Drawdown",
                        f"{risk_report.get('current_drawdown', 0):.2%}"
                    )
                
                # Risk warnings
                if risk_report.get('warnings'):
                    st.warning("Risk Warnings:")
                    for warning in risk_report['warnings']:
                        st.write(f"⚠️ {warning}")

if __name__ == "__main__":
    main()
