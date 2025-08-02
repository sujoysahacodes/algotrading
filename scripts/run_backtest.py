#!/usr/bin/env python3
"""
Backtesting Script

Run backtests for different strategies on historical data.
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.manager import data_manager
from strategies import EMAAdxStrategy, ZScoreMeanReversionStrategy, FXTrendStrategy
from backtesting import BacktestEngine
from utils import config, setup_logging, get_logger

# Setup logging
setup_logging(level="INFO", log_file="logs/backtest.log")
logger = get_logger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run algorithmic trading backtests')
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['ema_adx', 'zscore', 'fx_trend', 'all'],
        default='all',
        help='Strategy to backtest'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Trading symbol'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default='2024-01-01',
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--frequency',
        type=str,
        default='1d',
        help='Data frequency'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000,
        help='Initial capital'
    )
    
    return parser.parse_args()

def create_strategies():
    """Create strategy instances."""
    strategies = {}
    
    # EMA-ADX Strategy
    ema_config = config.get('strategies.trend_following.ema_adx', {})
    strategies['ema_adx'] = EMAAdxStrategy(ema_config)
    
    # Z-Score Mean Reversion Strategy
    zscore_config = config.get('strategies.mean_reversion.zscore', {})
    strategies['zscore'] = ZScoreMeanReversionStrategy(zscore_config)
    
    # FX Trend Strategy
    fx_config = config.get('strategies.trend_following.fx_trend', {
        'short_interval': '30s',
        'long_interval': '5min',
        'trend_threshold': 0.02
    })
    strategies['fx_trend'] = FXTrendStrategy(fx_config)
    
    return strategies

def main():
    """Main backtesting function."""
    args = parse_arguments()
    
    logger.info(f"Starting backtests for {args.symbol} from {args.start} to {args.end}")
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # Fetch data
    logger.info(f"Fetching data for {args.symbol}...")
    data = data_manager.get_historical_data(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        frequency=args.frequency
    )
    
    if data.empty:
        logger.error(f"No data found for {args.symbol}")
        return
    
    logger.info(f"Retrieved {len(data)} data points")
    
    # Initialize backtest engine
    backtest_config = {
        'initial_capital': args.initial_capital,
        'commission': 0.001,
        'slippage': 0.0005
    }
    engine = BacktestEngine(backtest_config)
    
    # Create strategies
    strategies = create_strategies()
    
    # Run backtests
    if args.strategy == 'all':
        strategies_to_test = list(strategies.values())
    else:
        strategies_to_test = [strategies[args.strategy]]
    
    results = []
    
    for strategy in strategies_to_test:
        logger.info(f"Running backtest for {strategy.name}...")
        
        try:
            result = engine.run_backtest(
                strategy=strategy,
                data=data,
                symbol=args.symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            results.append(result)
            
            # Print results
            print(f"\n{'='*50}")
            print(f"Strategy: {result.strategy_name}")
            print(f"Symbol: {result.symbol}")
            print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
            print(f"{'='*50}")
            print(f"Initial Capital: ${result.initial_capital:,.2f}")
            print(f"Final Value: ${result.portfolio_values[-1]:,.2f}" if result.portfolio_values else "N/A")
            print(f"Total Return: {result.total_return:.2%}")
            print(f"Annualized Return: {result.annualized_return:.2%}")
            print(f"Volatility: {result.volatility:.2%}")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {result.max_drawdown:.2%}")
            print(f"Win Rate: {result.win_rate:.1%}")
            print(f"Number of Trades: {result.num_trades}")
            print(f"Avg Trade Return: {result.avg_trade_return:.2%}")
            
        except Exception as e:
            logger.error(f"Error running backtest for {strategy.name}: {e}")
    
    # Compare strategies if multiple
    if len(strategies_to_test) > 1:
        logger.info("Comparing strategies...")
        comparison = engine.compare_strategies(strategies_to_test, data, args.symbol)
        
        print(f"\n{'='*80}")
        print("STRATEGY COMPARISON")
        print(f"{'='*80}")
        print(comparison.to_string(index=False, float_format='%.3f'))
    
    logger.info("Backtesting completed")

if __name__ == '__main__':
    main()
