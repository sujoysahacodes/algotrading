"""
Configuration Management Module
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for the algorithmic trading system."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Replace environment variables
            config = self._replace_env_vars(config)
            return config
        except FileNotFoundError:
            # Return default configuration if file not found
            return self._get_default_config()
    
    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace environment variables in configuration."""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'sources': {
                    'yahoo': {'enabled': True},
                    'openbb': {'enabled': False, 'api_key': '${OPENBB_API_KEY}'},
                    'alpaca': {'enabled': False, 'api_key': '${ALPACA_API_KEY}', 'secret_key': '${ALPACA_SECRET_KEY}'}
                },
                'storage': {
                    'type': 'sqlite',
                    'path': 'data/market_data.db'
                }
            },
            'strategies': {
                'trend_following': {
                    'ema_adx': {
                        'ema_fast': 12,
                        'ema_slow': 26,
                        'adx_period': 14,
                        'adx_threshold': 25
                    },
                    'macd': {
                        'fast_period': 12,
                        'slow_period': 26,
                        'signal_period': 9
                    }
                },
                'mean_reversion': {
                    'zscore': {
                        'lookback_window': 20,
                        'entry_threshold': 2.0,
                        'exit_threshold': 0.5
                    }
                }
            },
            'risk': {
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'max_portfolio_risk': 0.1,
                'var_confidence': 0.95
            },
            'brokers': {
                'alpaca': {
                    'base_url': 'https://paper-api.alpaca.markets',
                    'api_key': '${ALPACA_API_KEY}',
                    'secret_key': '${ALPACA_SECRET_KEY}'
                },
                'ib': {
                    'host': 'localhost',
                    'port': 7497,
                    'client_id': 1
                }
            },
            'backtesting': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

# Global configuration instance
config = Config()
