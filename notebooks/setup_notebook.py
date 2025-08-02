#!/usr/bin/env python3
"""
Notebook setup helper - Run this before starting Jupyter
"""

import sys
import os
from pathlib import Path

def setup_notebook_environment():
    """Setup the Python path for notebook execution"""
    
    # Get the current directory
    current_dir = Path.cwd()
    
    # If we're in the notebooks directory, go up one level
    if current_dir.name == 'notebooks':
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    # Add src directory to Python path
    src_path = project_root / 'src'
    
    if src_path.exists():
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    else:
        print(f"❌ Source directory not found: {src_path}")
        return False
    
    # Test imports
    try:
        from data import data_manager
        from strategies import EMAAdxStrategy, ZScoreMeanReversionStrategy, FXTrendStrategy
        from backtesting import BacktestEngine
        from risk import RiskManager
        from utils import setup_logging
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up notebook environment...")
    
    if setup_notebook_environment():
        print("\n🎉 Environment setup complete!")
        print("\nNow you can:")
        print("1. Run: jupyter notebook")
        print("2. Open: notebooks/Strategy_Analysis.ipynb")
    else:
        print("\n❌ Environment setup failed!")
        print("\nTroubleshooting steps:")
        print("1. Make sure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Install the package: pip install -e .")
