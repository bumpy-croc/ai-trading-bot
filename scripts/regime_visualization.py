#!/usr/bin/env python3
"""
Market Regime Visualization Script

This script fetches historical price data from Binance and applies the regime detector
to visualize how well the regime predictions align with actual market movements.

Usage:
    python scripts/regime_visualization.py --symbol BTCUSDT --timeframe 1h --days 30
    python scripts/regime_visualization.py --symbol ETHUSDT --timeframe 4h --days 90
"""

import argparse
import logging

# Add src to path for imports
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime.detector import RegimeDetector

from data_providers.binance_provider import BinanceProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_price_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch historical price data from Binance"""
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")
    
    # Initialize Binance provider (public mode, no credentials needed for historical data)
    provider = BinanceProvider()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch data
    df = provider.get_historical_data(symbol, timeframe, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol} {timeframe}")
    
    logger.info(f"Fetched {len(df)} candles from {df.index.min()} to {df.index.max()}")
    return df


def apply_regime_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Apply regime detection to price data"""
    logger.info("Applying regime detection")
    
    # Initialize regime detector with default config
    detector = RegimeDetector()
    
    # Annotate the dataframe with regime predictions
    df_with_regime = detector.annotate(df)
    
    logger.info("Regime detection completed")
    return df_with_regime


def create_visualization(df: pd.DataFrame, symbol: str, timeframe: str, days: int) -> None:
    """Create comprehensive visualization of price data and regime predictions"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    # Main price chart with regime overlays
    ax1 = fig.add_subplot(gs[0])
    
    # Plot price data
    ax1.plot(df.index, df['close'], 'k-', linewidth=1, alpha=0.8, label='Price')
    
    # Color regions based on regime
    regime_colors = {
        'trend_up:low_vol': 'lightgreen',
        'trend_up:high_vol': 'green', 
        'trend_down:low_vol': 'lightcoral',
        'trend_down:high_vol': 'red',
        'range:low_vol': 'lightblue',
        'range:high_vol': 'blue'
    }
    
    # Create regime background regions
    current_regime = None
    regime_start = None
    
    for _i, (timestamp, row) in enumerate(df.iterrows()):
        regime = row['regime_label']
        
        if regime != current_regime:
            # Close previous regime region
            if current_regime is not None and regime_start is not None:
                color = regime_colors.get(current_regime, 'lightgray')
                ax1.axvspan(regime_start, timestamp, alpha=0.3, color=color)
            
            # Start new regime region
            current_regime = regime
            regime_start = timestamp
    
    # Close final regime region
    if current_regime is not None and regime_start is not None:
        color = regime_colors.get(current_regime, 'lightgray')
        ax1.axvspan(regime_start, df.index[-1], alpha=0.3, color=color)
    
    ax1.set_title(f'{symbol} Price with Market Regime Detection\n{timeframe} timeframe, last {days} days', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add regime labels on the chart
    regime_changes = df[df['regime_label'] != df['regime_label'].shift(1)]
    for timestamp, row in regime_changes.iterrows():
        ax1.axvline(x=timestamp, color='black', linestyle='--', alpha=0.5)
        ax1.text(timestamp, ax1.get_ylim()[1] * 0.95, 
                row['regime_label'].replace(':', '\n'), 
                rotation=90, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Trend score subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['trend_score'], 'purple', linewidth=1, label='Trend Score')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.fill_between(df.index, df['trend_score'], 0, 
                     where=(df['trend_score'] >= 0), color='green', alpha=0.3, label='Positive Trend')
    ax2.fill_between(df.index, df['trend_score'], 0, 
                     where=(df['trend_score'] < 0), color='red', alpha=0.3, label='Negative Trend')
    ax2.set_ylabel('Trend Score', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Volatility (ATR percentile) subplot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['atr_percentile'], 'orange', linewidth=1, label='ATR Percentile')
    ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Vol Threshold')
    ax3.fill_between(df.index, 0, 1, 
                     where=(df['vol_label'] == 'high_vol'), color='red', alpha=0.2, label='High Vol')
    ax3.fill_between(df.index, 0, 1, 
                     where=(df['vol_label'] == 'low_vol'), color='green', alpha=0.2, label='Low Vol')
    ax3.set_ylabel('Volatility\n(ATR %ile)', fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # Regime confidence subplot
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, df['regime_confidence'], 'blue', linewidth=1, label='Regime Confidence')
    ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
    ax4.fill_between(df.index, 0, df['regime_confidence'], alpha=0.3, color='blue')
    ax4.set_ylabel('Confidence', fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    # Add regime legend
    legend_elements = []
    for regime, color in regime_colors.items():
        legend_elements.append(patches.Patch(color=color, alpha=0.3, label=regime))
    
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8, 
              title='Market Regimes', title_fontsize=9)
    
    # Add statistics text box
    stats_text = f"""Regime Statistics:
• Total periods: {len(df)}
• Trend Up: {len(df[df['trend_label'] == 'trend_up'])} ({len(df[df['trend_label'] == 'trend_up'])/len(df)*100:.1f}%)
• Trend Down: {len(df[df['trend_label'] == 'trend_down'])} ({len(df[df['trend_label'] == 'trend_down'])/len(df)*100:.1f}%)
• Range: {len(df[df['trend_label'] == 'range'])} ({len(df[df['trend_label'] == 'range'])/len(df)*100:.1f}%)
• High Vol: {len(df[df['vol_label'] == 'high_vol'])} ({len(df[df['vol_label'] == 'high_vol'])/len(df)*100:.1f}%)
• Avg Confidence: {df['regime_confidence'].mean():.3f}"""
    
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"regime_analysis_{symbol}_{timeframe}_{days}d.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved as {output_file}")
    
    # Show the plot
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize market regime detection accuracy')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze (default: 30)')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    try:
        # Fetch price data
        df = fetch_price_data(args.symbol, args.timeframe, args.days)
        
        # Apply regime detection
        df_with_regime = apply_regime_detection(df)
        
        # Create visualization
        create_visualization(df_with_regime, args.symbol, args.timeframe, args.days)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

