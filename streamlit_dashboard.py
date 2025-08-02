"""
Simplified Algorithmic Trading Dashboard

A Streamlit-based dashboard for monitoring trading strategies,
backtesting results, and risk metrics using simplified modules.
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our simplified trading modules
from simple_trading import data_manager, SimpleTradingStrategy, SimpleBacktester

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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #26C281;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Strategy Classes
class EMAStrategy(SimpleTradingStrategy):
    """EMA Crossover Strategy"""
    
    def __init__(self, config):
        super().__init__(config)
        self.fast = config.get('ema_fast', 12)
        self.slow = config.get('ema_slow', 26)
    
    def calculate_indicators(self, data):
        indicators = data.copy()
        indicators['ema_fast'] = self.calculate_ema(data['close'], self.fast)
        indicators['ema_slow'] = self.calculate_ema(data['close'], self.slow)
        
        # Generate signals
        indicators['signal'] = 0
        indicators.loc[indicators['ema_fast'] > indicators['ema_slow'], 'signal'] = 1
        indicators.loc[indicators['ema_fast'] < indicators['ema_slow'], 'signal'] = -1
        
        return indicators

class ZScoreStrategy(SimpleTradingStrategy):
    """Z-Score Mean Reversion Strategy"""
    
    def __init__(self, config):
        super().__init__(config)
        self.window = config.get('lookback_window', 20)
        self.threshold = config.get('entry_threshold', 2.0)
    
    def calculate_indicators(self, data):
        indicators = data.copy()
        indicators['rolling_mean'] = data['close'].rolling(self.window).mean()
        indicators['rolling_std'] = data['close'].rolling(self.window).std()
        indicators['zscore'] = (data['close'] - indicators['rolling_mean']) / indicators['rolling_std']
        
        # Bollinger Bands
        indicators['bb_upper'] = indicators['rolling_mean'] + (2 * indicators['rolling_std'])
        indicators['bb_lower'] = indicators['rolling_mean'] - (2 * indicators['rolling_std'])
        
        # Generate signals
        indicators['signal'] = 0
        indicators.loc[indicators['zscore'] < -self.threshold, 'signal'] = 1
        indicators.loc[indicators['zscore'] > self.threshold, 'signal'] = -1
        
        return indicators

@st.cache_data
def load_data(symbol, start_date, end_date):
    """Load historical data with caching"""
    try:
        data = data_manager.get_historical_data(symbol, start_date, end_date)
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_performance_metrics(data, signals):
    """Calculate performance metrics"""
    if data.empty or len(signals) == 0:
        return {}
    
    returns = data['close'].pct_change()
    strategy_returns = returns * pd.Series(signals, index=data.index).shift(1)
    
    # Calculate metrics
    total_return = (1 + strategy_returns.dropna()).prod() - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = total_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + strategy_returns.fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_trades = (strategy_returns > 0).sum()
    total_trades = (strategy_returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades
    }

def main():
    """Main dashboard function"""
    
    # Header
    st.title("📈 Algorithmic Trading Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Symbol",
        options=['AAPL', 'MSFT', 'GOOGL', 'SPY', 'TSLA', 'AMZN', 'NVDA'],
        index=0
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 1, 1))
    
    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Select Strategy",
        options=['EMA Crossover', 'Z-Score Mean Reversion'],
        index=0
    )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    if strategy_type == 'EMA Crossover':
        ema_fast = st.sidebar.slider("Fast EMA", 5, 50, 12)
        ema_slow = st.sidebar.slider("Slow EMA", 10, 100, 26)
        config = {'ema_fast': ema_fast, 'ema_slow': ema_slow}
        strategy = EMAStrategy(config)
    else:
        lookback = st.sidebar.slider("Lookback Window", 10, 50, 20)
        threshold = st.sidebar.slider("Entry Threshold", 1.0, 3.0, 2.0, 0.1)
        config = {'lookback_window': lookback, 'entry_threshold': threshold}
        strategy = ZScoreStrategy(config)
    
    # Load data button
    if st.sidebar.button("🔄 Load Data & Run Analysis", type="primary"):
        
        # Load data
        with st.spinner(f"Loading {symbol} data..."):
            data = load_data(symbol, start_date, end_date)
        
        if data.empty:
            st.error("No data loaded. Please check your inputs.")
            return
        
        # Calculate indicators
        with st.spinner("Calculating strategy indicators..."):
            indicators = strategy.calculate_indicators(data)
        
        # Calculate performance
        signals = indicators['signal'].fillna(0)
        metrics = calculate_performance_metrics(data, signals)
        
        # Store in session state
        st.session_state.data = data
        st.session_state.indicators = indicators
        st.session_state.metrics = metrics
        st.session_state.symbol = symbol
        st.session_state.strategy_type = strategy_type
    
    # Display results if data exists
    if 'data' in st.session_state and not st.session_state.data.empty:
        
        data = st.session_state.data
        indicators = st.session_state.indicators
        metrics = st.session_state.metrics
        
        # Key Metrics
        st.header(f"📊 {st.session_state.symbol} - {st.session_state.strategy_type}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            return_color = "positive" if metrics.get('total_return', 0) > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Return</h4>
                <h2 class="{return_color}">{metrics.get('total_return', 0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Sharpe Ratio</h4>
                <h2 class="neutral">{metrics.get('sharpe_ratio', 0):.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Volatility</h4>
                <h2 class="neutral">{metrics.get('volatility', 0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            dd_color = "negative" if metrics.get('max_drawdown', 0) < 0 else "neutral"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Max Drawdown</h4>
                <h2 class="{dd_color}">{metrics.get('max_drawdown', 0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Win Rate</h4>
                <h2 class="neutral">{metrics.get('win_rate', 0):.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Signals", "📊 Indicators", "💰 Performance", "📋 Data"])
        
        with tab1:
            # Price chart with signals
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # Buy signals
            buy_signals = indicators[indicators['signal'] == 1]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ))
            
            # Sell signals
            sell_signals = indicators[indicators['signal'] == -1]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ))
            
            fig.update_layout(
                title=f"{st.session_state.symbol} Price with Trading Signals",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Strategy-specific indicators
            if st.session_state.strategy_type == 'EMA Crossover':
                
                fig = make_subplots(rows=2, cols=1, 
                                  subplot_titles=('Price with EMAs', 'EMA Signals'),
                                  vertical_spacing=0.1)
                
                # Price and EMAs
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['close'], 
                                       name='Close', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['ema_fast'], 
                                       name=f'EMA {config["ema_fast"]}', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['ema_slow'], 
                                       name=f'EMA {config["ema_slow"]}', line=dict(color='red')), row=1, col=1)
                
                # Signals
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['signal'], 
                                       name='Signal', line=dict(color='purple')), row=2, col=1)
                
            else:  # Z-Score strategy
                
                fig = make_subplots(rows=3, cols=1,
                                  subplot_titles=('Price with Bollinger Bands', 'Z-Score', 'Signals'),
                                  vertical_spacing=0.08)
                
                # Price with Bollinger Bands
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['close'], 
                                       name='Close', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['rolling_mean'], 
                                       name='Mean', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['bb_upper'], 
                                       name='Upper BB', line=dict(color='red', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['bb_lower'], 
                                       name='Lower BB', line=dict(color='red', dash='dash')), row=1, col=1)
                
                # Z-Score
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['zscore'], 
                                       name='Z-Score', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=config['entry_threshold'], line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=-config['entry_threshold'], line_dash="dash", line_color="red", row=2, col=1)
                
                # Signals
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['signal'], 
                                       name='Signal', line=dict(color='green')), row=3, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Performance analysis
            returns = data['close'].pct_change()
            strategy_returns = returns * indicators['signal'].shift(1)
            
            # Cumulative returns
            cum_returns = (1 + strategy_returns.fillna(0)).cumprod()
            cum_benchmark = (1 + returns.fillna(0)).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, 
                                   name='Strategy', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=cum_benchmark.index, y=cum_benchmark, 
                                   name='Buy & Hold', line=dict(color='blue')))
            
            fig.update_layout(
                title="Cumulative Returns Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strategy Performance")
                perf_data = {
                    'Metric': ['Total Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Total Trades'],
                    'Value': [
                        f"{metrics.get('total_return', 0):.2%}",
                        f"{metrics.get('volatility', 0):.2%}",
                        f"{metrics.get('sharpe_ratio', 0):.2f}",
                        f"{metrics.get('max_drawdown', 0):.2%}",
                        f"{metrics.get('win_rate', 0):.1%}",
                        f"{metrics.get('total_trades', 0)}"
                    ]
                }
                st.dataframe(pd.DataFrame(perf_data), hide_index=True)
            
            with col2:
                st.subheader("Buy & Hold Performance")
                bh_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
                bh_vol = returns.std() * np.sqrt(252)
                bh_sharpe = bh_return / bh_vol if bh_vol > 0 else 0
                
                bh_data = {
                    'Metric': ['Total Return', 'Annualized Volatility', 'Sharpe Ratio'],
                    'Value': [
                        f"{bh_return:.2%}",
                        f"{bh_vol:.2%}",
                        f"{bh_sharpe:.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(bh_data), hide_index=True)
        
        with tab4:
            # Raw data display
            st.subheader("Historical Data")
            st.dataframe(data.tail(100))
            
            st.subheader("Strategy Indicators")
            st.dataframe(indicators[['close', 'signal'] + [col for col in indicators.columns if col not in ['close', 'signal']]].tail(100))
    
    else:
        # Welcome message
        st.info("👆 Configure your strategy parameters in the sidebar and click 'Load Data & Run Analysis' to get started!")
        
        # Sample visualization
        st.subheader("📈 Sample Analysis")
        st.write("This dashboard provides:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📊 Real-time Data**")
            st.write("- Live market data from Yahoo Finance")
            st.write("- Multiple symbols supported")
            st.write("- Customizable date ranges")
        
        with col2:
            st.write("**🧠 Strategy Analysis**")
            st.write("- EMA Crossover strategy")
            st.write("- Z-Score mean reversion")
            st.write("- Configurable parameters")
        
        with col3:
            st.write("**📈 Performance Metrics**")
            st.write("- Total returns & Sharpe ratio")
            st.write("- Risk metrics & drawdown")
            st.write("- Interactive visualizations")

if __name__ == "__main__":
    main()
