"""
Minimal AlgoTrading Setup for Development
"""

from setuptools import setup, find_packages

setup(
    name="algotrading",
    version="1.0.0",
    description="Algorithmic Trading System for Trend-Following and Mean-Reversion Strategies",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "yfinance>=0.2.0",
        "matplotlib>=3.5.0",
        "plotly>=5.10.0",
        "streamlit>=1.25.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "requests>=2.28.0",
        "ta>=0.10.0",
    ],
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
