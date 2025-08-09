#!/usr/bin/env python3
"""
Live Sentiment Trading Demo

This script demonstrates how sentiment analysis works in real-time trading scenarios.
It shows the difference between historical and live sentiment data and how trading
decisions are made with fresh sentiment information.

Usage:
    python examples/live_sentiment_demo.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, Any

from data_providers.senticrypt_provider import SentiCryptProvider
from strategies.ml_with_sentiment import MlWithSentiment

from utils.logging_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

class LiveSentimentDemo:
    """Demonstrates live sentiment analysis in trading scenarios"""
    
    def __init__(self):
        self.sentiment_provider = None
        self.strategy = None
        self.demo_data = None
        
    def setup_demo(self):
        """Initialize the demo components"""
        print("🚀 Setting up Live Sentiment Trading Demo...")
        
        # Initialize sentiment provider in live mode
        try:
            self.sentiment_provider = SentiCryptProvider(
                csv_path='data/senticrypt_sentiment_data.csv',
                live_mode=True,
                cache_duration_minutes=5  # Refresh every 5 minutes for demo
            )
            print(f"✅ Sentiment provider initialized with {len(self.sentiment_provider.data)} historical records")
        except Exception as e:
            print(f"❌ Failed to initialize sentiment provider: {e}")
            return False
        
        # Initialize ML strategy
        try:
            self.strategy = MlWithSentiment(
                name="LiveSentimentDemo",
                use_sentiment=True,
                sentiment_csv_path='data/senticrypt_sentiment_data.csv'
            )
            print("✅ ML Sentiment Strategy initialized")
        except Exception as e:
            print(f"❌ Failed to initialize strategy: {e}")
            return False
        
        # Create demo price data (simulating recent price movements)
        self.create_demo_data()
        
        return True
    
    def create_demo_data(self):
        """Create realistic demo price data for the last few hours"""
        print("📊 Creating demo price data...")
        
        # Generate hourly data for the last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        # Create time index
        time_index = pd.date_range(start=start_time, end=end_time, freq='1H')
        
        # Generate realistic BTC price data (random walk around current price)
        np.random.seed(42)  # For reproducible demo
        base_price = 95000  # Approximate BTC price
        
        # Generate price movements
        returns = np.random.normal(0, 0.02, len(time_index))  # 2% hourly volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create OHLCV data
        self.demo_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, len(time_index))
        }, index=time_index)
        
        print(f"✅ Created demo data: {len(self.demo_data)} hourly candles")
        print(f"   Time range: {self.demo_data.index.min()} to {self.demo_data.index.max()}")
        print(f"   Price range: ${self.demo_data['close'].min():.2f} - ${self.demo_data['close'].max():.2f}")
    
    def demonstrate_sentiment_comparison(self):
        """Show the difference between historical and live sentiment"""
        print("\n" + "="*60)
        print("🔍 SENTIMENT DATA COMPARISON")
        print("="*60)
        
        # Get historical sentiment for yesterday
        yesterday = datetime.now() - timedelta(days=1)
        historical_sentiment = self.sentiment_provider.get_sentiment_for_date(yesterday, use_live=False)
        
        print(f"📚 Historical Sentiment (24h ago):")
        for key, value in historical_sentiment.items():
            print(f"   {key}: {value:.4f}")
        
        # Get live sentiment
        print(f"\n🔴 Live Sentiment (Current):")
        try:
            live_sentiment = self.sentiment_provider.get_live_sentiment(force_refresh=True)
            for key, value in live_sentiment.items():
                print(f"   {key}: {value:.4f}")
            
            # Calculate differences
            print(f"\n📊 Sentiment Changes:")
            for key in live_sentiment.keys():
                if key in historical_sentiment:
                    change = live_sentiment[key] - historical_sentiment[key]
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f"   {key}: {change:+.4f} {direction}")
        
        except Exception as e:
            print(f"❌ Could not fetch live sentiment: {e}")
            print("   Using cached or historical data...")
    
    def simulate_live_trading_decisions(self):
        """Simulate how trading decisions are made with live sentiment"""
        print("\n" + "="*60)
        print("🤖 LIVE TRADING SIMULATION")
        print("="*60)
        
        # Process the demo data with the strategy
        processed_data = self.strategy.calculate_indicators(self.demo_data)
        
        # Show the last few data points to demonstrate live vs historical sentiment
        print("\n📊 Recent Data Points (Last 5 hours):")
        print("-" * 80)
        
        recent_data = processed_data.tail(5)
        for idx, (timestamp, row) in enumerate(recent_data.iterrows()):
            print(f"\n⏰ {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   💰 Price: ${row['close']:.2f}")
            
            if not pd.isna(row.get('ml_prediction')):
                print(f"   🎯 ML Prediction: ${row['ml_prediction']:.2f}")
                print(f"   📊 Confidence: {row.get('prediction_confidence', 0):.3f}")
            
            sentiment_freshness = row.get('sentiment_freshness', -1)
            if sentiment_freshness > 0:
                print(f"   🔴 LIVE SENTIMENT:")
            elif sentiment_freshness == 0:
                print(f"   📚 HISTORICAL SENTIMENT:")
            else:
                print(f"   ❌ NO SENTIMENT DATA:")
            
            if not pd.isna(row.get('sentiment_primary')):
                print(f"      Primary: {row['sentiment_primary']:.4f}")
                print(f"      Momentum: {row.get('sentiment_momentum', 0):.4f}")
                print(f"      Volatility: {row.get('sentiment_volatility', 0):.4f}")
            
            # Check trading signals
            if idx >= 1:  # Need at least 2 points for signal
                entry_signal = self.strategy.check_entry_conditions(processed_data, len(processed_data) - 5 + idx)
                if entry_signal:
                    print(f"   🚀 ENTRY SIGNAL DETECTED!")
                else:
                    print(f"   ⏸️  No entry signal")
    
    def demonstrate_api_freshness(self):
        """Show how API calls provide fresh sentiment data"""
        print("\n" + "="*60)
        print("🌐 API FRESHNESS DEMONSTRATION")
        print("="*60)
        
        print("Making multiple API calls to show data freshness...")
        
        for i in range(3):
            print(f"\n🔄 API Call #{i+1}:")
            try:
                # Force refresh to get latest data
                start_time = time.time()
                sentiment = self.sentiment_provider.get_live_sentiment(force_refresh=True)
                api_time = time.time() - start_time
                
                print(f"   ⏱️  API Response Time: {api_time:.2f}s")
                print(f"   📊 Primary Sentiment: {sentiment.get('sentiment_primary', 0):.4f}")
                print(f"   📈 Momentum: {sentiment.get('sentiment_momentum', 0):.4f}")
                
                # Show cache status
                cache_age = (datetime.now() - self.sentiment_provider._last_api_call).total_seconds()
                print(f"   🗄️  Cache Age: {cache_age:.1f}s")
                
            except Exception as e:
                print(f"   ❌ API Error: {e}")
            
            if i < 2:  # Don't sleep after last iteration
                print("   ⏳ Waiting 10 seconds...")
                time.sleep(10)
    
    def show_trading_advantages(self):
        """Explain the advantages of live sentiment in trading"""
        print("\n" + "="*60)
        print("💡 LIVE SENTIMENT TRADING ADVANTAGES")
        print("="*60)
        
        advantages = [
            "🎯 Real-time market sentiment capture",
            "⚡ Faster reaction to sentiment shifts", 
            "📈 Improved entry/exit timing",
            "🛡️ Better risk management with fresh data",
            "🔄 Adaptive to breaking news and events",
            "📊 Higher prediction accuracy",
            "💰 Potential for better returns"
        ]
        
        for advantage in advantages:
            print(f"   {advantage}")
        
        print(f"\n🔧 Technical Implementation:")
        print(f"   • API calls every 15 minutes (configurable)")
        print(f"   • Intelligent caching to avoid rate limits")
        print(f"   • Fallback to historical data if API fails")
        print(f"   • Sentiment freshness scoring")
        print(f"   • Confidence boosting for fresh sentiment")
    
    def show_potential_challenges(self):
        """Discuss challenges and solutions"""
        print("\n" + "="*60)
        print("⚠️  CHALLENGES & SOLUTIONS")
        print("="*60)
        
        challenges = [
            ("🌐 API Rate Limits", "Smart caching with configurable refresh intervals"),
            ("📡 Network Latency", "Timeout handling and fallback mechanisms"),
            ("💾 Data Freshness", "Timestamp tracking and age-based weighting"),
            ("🔄 API Failures", "Graceful degradation to historical data"),
            ("💰 API Costs", "Efficient caching reduces unnecessary calls"),
            ("🎯 False Signals", "Confidence scoring and multi-factor validation")
        ]
        
        for challenge, solution in challenges:
            print(f"   {challenge}")
            print(f"      → {solution}")
            print()
    
    def run_complete_demo(self):
        """Run the complete live sentiment demo"""
        print("🎭 LIVE SENTIMENT TRADING ANALYSIS DEMO")
        print("=" * 60)
        
        if not self.setup_demo():
            print("❌ Demo setup failed!")
            return
        
        try:
            # Run all demo sections
            self.demonstrate_sentiment_comparison()
            self.simulate_live_trading_decisions()
            self.demonstrate_api_freshness()
            self.show_trading_advantages()
            self.show_potential_challenges()
            
            print("\n" + "="*60)
            print("✅ DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\n🎯 Key Takeaways:")
            print("   • Live sentiment provides fresher, more actionable data")
            print("   • Real-time API calls enable faster market reaction")
            print("   • Intelligent caching balances freshness with efficiency")
            print("   • Fallback mechanisms ensure robust operation")
            print("   • Sentiment freshness scoring improves decision quality")
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            logger.exception("Demo failed")

def main():
    """Main demo function"""
    demo = LiveSentimentDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main() 