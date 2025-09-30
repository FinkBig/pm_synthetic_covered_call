"""
Simplified portfolio strategies for covered calls analysis.
Contains 3 core strategies: Vanilla BTC, Covered Calls, and PM Covered Call replication.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_metrics(returns_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate common portfolio metrics"""
    if returns_df.empty:
        return {}
        
    # Calculate daily returns
    daily_returns = returns_df['portfolio_value'].pct_change().dropna()
    
    # Basic metrics
    total_return = (returns_df['portfolio_value'].iloc[-1] / returns_df['portfolio_value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (365 / len(returns_df)) - 1
    volatility = daily_returns.std() * np.sqrt(365)
    
    # Drawdown calculation
    peak = returns_df['portfolio_value'].cummax()
    drawdown = (returns_df['portfolio_value'] - peak) / peak
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def vanilla_btc_strategy(price_data: pd.DataFrame, initial_capital: float = 10000.0, fees_pct: float = 0.0) -> pd.DataFrame:
    """Simple buy and hold BTC strategy"""
    if price_data.empty:
        return pd.DataFrame()
        
    # Buy BTC at first price (no fees for comparability)
    first_price = price_data['close'].iloc[0]  # Use close price for consistency
    btc_shares = initial_capital / first_price
    
    # Calculate portfolio value over time
    portfolio_values = []
    for _, row in price_data.iterrows():
        portfolio_value = btc_shares * row['close']
        portfolio_values.append({
            'timestamp': row['timestamp'],
            'portfolio_value': portfolio_value,
            'btc_price': row['close']
        })
        
    return pd.DataFrame(portfolio_values)

def covered_call_strategy(price_data: pd.DataFrame, options_data: pd.DataFrame, 
                         initial_capital: float = 10000.0, fees_pct: float = 0.0, 
                         otm_pct: float = 0.02, allocation_pct: float = 0.10) -> pd.DataFrame:
    """Covered call strategy selling +2% OTM calls daily with dynamic 10% allocation"""
    if price_data.empty or options_data.empty:
        return pd.DataFrame()
        
    # Merge price and options data
    merged_data = pd.merge(
        price_data, 
        options_data, 
        left_on=price_data['timestamp'].dt.date, 
        right_on=pd.to_datetime(options_data['date']).dt.date,
        how='left'
    )
    
    # Initial allocation: 90% BTC, 10% for options positions (no fees for comparability)
    first_price = merged_data['close'].iloc[0]  # Use close price for consistency
    btc_capital = initial_capital * (1 - allocation_pct)
    options_capital = initial_capital * allocation_pct
    
    btc_shares = btc_capital / first_price
    cash = options_capital  # Initial cash for options positions
    
    portfolio_values = []
    
    for _, row in merged_data.iterrows():
        if pd.isna(row['premium_received']) or row['premium_received'] == 0:
            # No options data or no trades, just hold BTC and reinvest any cash
            if cash > 0:
                btc_to_buy = cash / row['close']
                btc_shares += btc_to_buy
                cash = 0.0
            portfolio_value = btc_shares * row['close'] + cash
            premium_received = 0
        else:
            # Calculate covered call returns using real Deribit data
            # Premium received is already the total USD amount from real Deribit data
            premium_received = row['premium_received']
            
            # Check if option is exercised using real Deribit data
            option_exercised = row['option_exercised']
            strike = row['strike_price']
            spot_close = row['close']
            
            # Calculate options position returns
            if option_exercised:
                # Option exercised - we need to sell BTC at strike price
                # The premium received represents the total premium for the position
                # We need to calculate how much BTC to sell based on the premium received
                # and the strike price
                btc_to_sell = premium_received / strike  # Approximate BTC to sell
                proceeds = btc_to_sell * strike
                cash += proceeds + premium_received
                btc_shares -= btc_to_sell
            else:
                # Option expires worthless - we keep the premium
                cash += premium_received
            
            # Reinvest options earnings into BTC first
            if cash > 0:
                btc_to_buy = cash / spot_close
                btc_shares += btc_to_buy
                cash = 0.0
            
            # Calculate current portfolio value
            portfolio_value = btc_shares * spot_close + cash
            
            # Dynamic 10% allocation: Take 10% of NEW portfolio value for next options position
            # FIX: Fund the options allocation by selling BTC shares (no free cash)
            cash_needed = portfolio_value * allocation_pct
            if cash_needed > 0:
                btc_to_sell = cash_needed / spot_close
                btc_shares -= btc_to_sell
                cash = cash_needed
            else:
                cash = 0.0
            
            # Update portfolio value (should remain ~same, just rebalanced)
            portfolio_value = btc_shares * spot_close + cash
        
        portfolio_values.append({
            'timestamp': row['timestamp'],
            'portfolio_value': portfolio_value,
            'btc_price': row['close'],
            'btc_value': btc_shares * row['close'],
            'options_value': cash,  # Cash represents options allocation for next day
            'premium_received': premium_received,
            'options_allocation_pct': (cash / portfolio_value) * 100 if portfolio_value > 0 else 0
        })
        
    return pd.DataFrame(portfolio_values)

def pm_covered_call_strategy(price_data: pd.DataFrame, pm_data: pd.DataFrame,
                           initial_capital: float = 10000.0, fees_pct: float = 0.0, 
                           allocation_pct: float = 0.10) -> pd.DataFrame:
    """PM covered call replication: Hold BTC + buy NO shares on higher strikes with daily rolling"""
    if price_data.empty or pm_data.empty:
        return pd.DataFrame()
        
    # Filter PM data for Level 3 (higher strikes) - this replicates covered calls
    pm_level3 = pm_data[pm_data['price_lvl'] == 3].copy()
    if pm_level3.empty:
        logger.warning("No Level 3 PM data found for covered call replication")
        return pd.DataFrame()
    
    # Merge price and PM data
    merged_data = pd.merge(
        price_data, 
        pm_level3, 
        left_on=price_data['timestamp'].dt.date, 
        right_on=pd.to_datetime(pm_level3['created_at']).dt.date,
        how='left'
    )
    
    # Initial allocation: 90% BTC, 10% for PM positions (no fees for comparability)
    first_price = merged_data['close'].iloc[0]  # Use close price for consistency
    btc_capital = initial_capital * (1 - allocation_pct)
    pm_capital = initial_capital * allocation_pct
    
    btc_shares = btc_capital / first_price
    cash = pm_capital
    
    # Dynamic 10% allocation - will be calculated each day based on portfolio value
    
    portfolio_values = []
    
    for _, row in merged_data.iterrows():
        if pd.isna(row.get('yes_price')) or pd.isna(row.get('strike_price')):
            # No PM data, just hold BTC and reinvest any cash
            if cash > 0:
                btc_to_buy = cash / row['close']
                btc_shares += btc_to_buy
                cash = 0.0
            portfolio_value = btc_shares * row['close'] + cash
        else:
            # Calculate PM position returns
            yes_price = row['yes_price']
            no_price = 1 - yes_price  # Calculate no_price from yes_price
            strike = row['strike_price']
            spot_close = row['close']
            
            # Buy NO shares with available cash
            if cash > 0:
                no_shares = cash / no_price
                cash = 0.0
            else:
                no_shares = 0.0
            
            # Calculate PM payout based on resolution from PM data
            # Resolution "NO" means BTC stayed below strike (we win)
            # Resolution "YES" means BTC went above strike (we lose)
            pm_payout = 1.0 if row.get('resolution') == 'NO' else 0.0
            pm_value = no_shares * pm_payout
            
            # Reinvest PM earnings into BTC first
            if pm_value > 0:
                btc_to_buy = pm_value / spot_close
                btc_shares += btc_to_buy
                pm_value = 0.0
            
            # Calculate current portfolio value
            portfolio_value = btc_shares * spot_close + pm_value
            
            # Dynamic 10% allocation: Take 10% of NEW portfolio value for next PM position
            # FIX: Fund the PM allocation by selling BTC shares (no free cash)
            cash_needed = portfolio_value * allocation_pct
            if cash_needed > 0:
                btc_to_sell = cash_needed / spot_close
                btc_shares -= btc_to_sell
                cash = cash_needed
            else:
                cash = 0.0
            
            # Update portfolio value (should remain ~same, just rebalanced)
            portfolio_value = btc_shares * spot_close + cash
        
        portfolio_values.append({
            'timestamp': row['timestamp'],
            'portfolio_value': portfolio_value,
            'btc_price': row['close'],
            'btc_value': btc_shares * row['close'],
            'pm_value': cash,  # Cash represents PM allocation for next day
            'pm_allocation_pct': (cash / portfolio_value) * 100 if portfolio_value > 0 else 0
        })
        
    return pd.DataFrame(portfolio_values)
