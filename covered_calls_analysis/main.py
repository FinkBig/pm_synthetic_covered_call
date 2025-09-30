"""
Simplified Covered Calls Analysis - Main Script
Compares 3 strategies: Vanilla BTC, Covered Calls, and PM Covered Call replication.
"""

import pandas as pd
import yaml
import logging
import os
from strategies import vanilla_btc_strategy, covered_call_strategy, pm_covered_call_strategy, calculate_metrics
from analysis import generate_charts, save_results, create_metrics_table

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """Load all required data files"""
    logger.info("Loading data files...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(__file__)
    
    # Load BTC price data
    btc_data = pd.read_csv(os.path.join(script_dir, 'data/btc_prices_daily.csv'), parse_dates=['timestamp'])
    logger.info(f"Loaded {len(btc_data)} days of BTC price data")
    
    # Load real Deribit options data
    options_data = pd.read_csv(os.path.join(script_dir, 'data/real_deribit_2pct_otm_data.csv'), parse_dates=['date'])
    logger.info(f"Loaded {len(options_data)} days of real Deribit options data")
    
    # Load PM data (filtered to exclude extreme outliers)
    pm_data = pd.read_csv(os.path.join(script_dir, 'data/limitless_pm_filtered.csv'), parse_dates=['created_at'])
    logger.info(f"Loaded {len(pm_data)} PM records (excluding extreme outliers)")
    
    return btc_data, options_data, pm_data

def main():
    """Main execution function"""
    logger.info("Starting Covered Calls Analysis...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Analysis period: {config['date_range']['start']} to {config['date_range']['end']}")
    
    # Create output directories
    script_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(script_dir, 'outputs/charts'), exist_ok=True)
    
    # Load data
    btc_data, options_data, pm_data = load_data()
    
    # Filter data by date range
    start_date = pd.to_datetime(config['date_range']['start'])
    end_date = pd.to_datetime(config['date_range']['end'])
    
    btc_data = btc_data[(btc_data['timestamp'] >= start_date) & (btc_data['timestamp'] <= end_date)]
    options_data = options_data[(pd.to_datetime(options_data['date']) >= start_date) & 
                               (pd.to_datetime(options_data['date']) <= end_date)]
    pm_data = pm_data[(pd.to_datetime(pm_data['created_at']) >= start_date) & 
                     (pd.to_datetime(pm_data['created_at']) <= end_date)]
    
    logger.info(f"Filtered data: {len(btc_data)} BTC days, {len(options_data)} options days, {len(pm_data)} PM records")
    
    # Calculate strategy returns
    logger.info("Calculating strategy returns...")
    
    # Strategy 1: Vanilla BTC (no fees for comparability)
    vanilla_returns = vanilla_btc_strategy(
        btc_data, 
        initial_capital=config['initial_capital'],
        fees_pct=0.0
    )
    logger.info(f"Vanilla BTC strategy: {len(vanilla_returns)} days")
    
    # Strategy 2: Covered Calls (no fees for comparability)
    covered_call_returns = covered_call_strategy(
        btc_data, 
        options_data,
        initial_capital=config['initial_capital'],
        fees_pct=0.0,
        allocation_pct=0.10
    )
    logger.info(f"Covered Call strategy: {len(covered_call_returns)} days")
    
    # Strategy 3: PM Covered Call replication (no fees for comparability)
    pm_returns = pm_covered_call_strategy(
        btc_data,
        pm_data,
        initial_capital=config['initial_capital'],
        fees_pct=0.0
    )
    logger.info(f"PM Covered Call strategy: {len(pm_returns)} days")
    
    # Prepare results
    strategy_returns = {
        'Vanilla BTC': vanilla_returns,
        'Covered Calls': covered_call_returns,
        'PM Covered Call': pm_returns
    }
    
    # Generate charts
    if config['generate_charts']:
        logger.info("Generating charts...")
        generate_charts(btc_data, strategy_returns, os.path.join(script_dir, 'outputs/charts'))
    
    # Save results
    if config['save_results']:
        logger.info("Saving results...")
        results_df = save_results(strategy_returns, os.path.join(script_dir, 'outputs/results.csv'))
        
        # Create comprehensive metrics table
        logger.info("Creating comprehensive metrics table...")
        metrics_df = create_metrics_table(strategy_returns, btc_data, os.path.join(script_dir, 'outputs'))
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("STRATEGY PERFORMANCE SUMMARY")
        logger.info("="*60)
        print(results_df.to_string(index=False))
        logger.info("="*60)
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()
