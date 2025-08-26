#!/usr/bin/env python3
"""
Data Commands for CLI

This module contains data-related functionality.
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
project_root = Path(__file__).parent.parent.parent
scripts_path = project_root / "scripts"
sys.path.insert(0, str(scripts_path))

from download_binance_data import download_data


def download_binance_data_wrapper(symbol, timeframe="1d", start_date=None, end_date=None, output_dir=None):
    """Wrapper for the download_binance_data function"""
    if output_dir is None:
        output_dir = project_root / "data"
    
    return download_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        output_dir=str(output_dir)
    )
