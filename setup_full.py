"""
AlgoTrading Setup Configuration
"""

from setuptools import setup, find_packages

setup(
    name="algotrading",
    version="1.0.0",
    description="Algorithmic Trading System for Trend-Following and Mean-Reversion Strategies",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core data processing
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        
        # Data sources
        "yfinance>=0.2.0",
        "openbb>=3.0.0",
        "alpaca-trade-api>=3.0.0",
        "ib-insync>=0.9.70",
        
        # Technical indicators
        "ta>=0.10.0",
        "talib-binary>=0.4.25",
        
        # Backtesting
        "backtrader>=1.9.76",
        "zipline-reloaded>=2.2.0",
        "vectorbt>=0.25.0",
        
        # API and networking
        "requests>=2.28.0",
        "websocket-client>=1.4.0",
        "asyncio>=3.4.3",
        "aiohttp>=3.8.0",
        
        # Database
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "timescaledb>=0.1.0",
        
        # Visualization and reporting
        "matplotlib>=3.5.0",
        "plotly>=5.10.0",
        "streamlit>=1.25.0",
        "dash>=2.10.0",
        
        # Risk management
        "pyfolio>=0.9.2",
        "quantlib>=1.29",
        
        # Configuration and logging
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "loguru>=0.6.0",
        
        # Docker and scheduling
        "schedule>=1.2.0",
        "python-crontab>=2.6.0",
        
        # Testing
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        
        # Jupyter for analysis
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "full": [
            "tensorflow>=2.10.0",
            "scikit-learn>=1.1.0",
            "xgboost>=1.6.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
