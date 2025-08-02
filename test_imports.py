#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"Project root: {project_root}")
print(f"Added to Python path: {src_path}")
print(f"Python path: {sys.path[:3]}...")

try:
    # Test imports
    print("\n🧪 Testing imports...")
    
    from data import data_manager
    print("✅ data_manager imported successfully")
    
    from strategies import EMAAdxStrategy, ZScoreMeanReversionStrategy, FXTrendStrategy
    print("✅ Strategy classes imported successfully")
    
    from backtesting import BacktestEngine
    print("✅ BacktestEngine imported successfully")
    
    from risk import RiskManager
    print("✅ RiskManager imported successfully")
    
    from utils import setup_logging
    print("✅ setup_logging imported successfully")
    
    print("\n🎉 All imports successful! You can now run the notebook.")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the project root directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Install the package in development mode: pip install -e .")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    sys.exit(1)
