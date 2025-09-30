#!/usr/bin/env python3
"""
Real Deribit Historical Data Fetcher
Fetches actual historical option trades for 2% OTM strikes based on BTC opening prices
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, timezone
import pytz
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DERIBIT_HISTORY_URL = "https://history.deribit.com/api/v2"
API_RETRY_ATTEMPTS = 3
API_RETRY_WAIT_S = 2
REQUEST_DELAY_S = 0.5
TRADES_COUNT_LIMIT = 1000
DERIBIT_EXPIRY_HOUR_UTC = 8

class DeribitApiError(Exception):
    """Custom exception for Deribit API errors."""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def make_api_request_with_retry(url: str, params: dict, retries: int = API_RETRY_ATTEMPTS, wait: float = API_RETRY_WAIT_S):
    """Makes API request with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            # Check for specific Deribit error structure
            if isinstance(data, dict) and 'error' in data:
                error_data = data['error']
                error_message = error_data.get('message', 'Unknown error') if isinstance(error_data, dict) else str(error_data)
                error_code = error_data.get('code', 'N/A') if isinstance(error_data, dict) else 'N/A'

                # Handle rate limit specifically
                if 'rate limit' in error_message.lower():
                    wait_time = wait * (2 ** attempt)
                    logger.warning(f"Rate limit hit (Attempt {attempt+1}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise DeribitApiError(f"Deribit API Error (Code: {error_code}): {error_message}", code=error_code)

            response.raise_for_status()
            return data
        except DeribitApiError as e:
            logger.error(f"Deribit API Error (Attempt {attempt+1}): {e}")
            if attempt + 1 >= retries: raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Network/Timeout error (Attempt {attempt+1}/{retries}): {e}.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (Attempt {attempt+1}/{retries}): {e}.")

        if attempt + 1 < retries:
            wait_time = wait * (2 ** attempt)
            logger.info(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
        else:
            logger.error("Max retries reached. Request failed.")
            if 'e' in locals(): raise e
            else: raise requests.exceptions.RequestException(f"Request failed after {retries} retries.")

    return None

def get_btc_price_from_deribit(target_date: datetime) -> Optional[float]:
    """Get BTC price from Deribit's own data at 08:00 UTC"""
    try:
        # Create 08:00 UTC timestamp for the target date
        target_0800_utc = target_date.replace(hour=8, minute=0, second=0, microsecond=0)
        target_0800_utc = target_0800_utc.replace(tzinfo=timezone.utc)
        
        # Convert to milliseconds
        start_ts = int(target_0800_utc.timestamp() * 1000)
        end_ts = start_ts + 3600000  # 1 hour window (08:00-09:00 UTC)
        
        params = {
            'instrument_name': 'BTC-PERPETUAL',
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
            'resolution': 60,  # 1 minute resolution
            'count': 60  # Max 60 minutes
        }
        
        url = f"{DERIBIT_HISTORY_URL}/public/get_tradingview_chart_data"
        data = make_api_request_with_retry(url, params)
        
        if not data or 'result' not in data or 'ticks' not in data['result']:
            logger.warning(f"No Deribit BTC price data for {target_date.strftime('%Y-%m-%d')} at 08:00 UTC")
            return None
            
        ticks = data['result']['ticks']
        if not ticks:
            logger.warning(f"No Deribit BTC price ticks for {target_date.strftime('%Y-%m-%d')} at 08:00 UTC")
            return None
        
        # Get the first tick (closest to 08:00 UTC)
        first_tick = ticks[0]
        if isinstance(first_tick, list) and len(first_tick) >= 2:
            # Format: [timestamp, open, high, low, close, volume]
            btc_price = float(first_tick[1])  # Open price
            logger.debug(f"Deribit BTC price at 08:00 UTC on {target_date.strftime('%Y-%m-%d')}: ${btc_price:,.2f}")
            return btc_price
        else:
            logger.warning(f"Unexpected tick format for {target_date.strftime('%Y-%m-%d')}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting Deribit BTC price for {target_date}: {e}")
        return None

def get_btc_price_from_existing_data(target_date: datetime) -> Optional[float]:
    """Get BTC price from our existing data (use open price as proxy for 08:00 UTC) - FALLBACK ONLY"""
    try:
        # Load existing BTC data
        btc_data = pd.read_csv('data/btc_prices_daily.csv', parse_dates=['timestamp'])
        
        # Find the row for the target date
        target_date_only = target_date.date()
        matching_rows = btc_data[btc_data['timestamp'].dt.date == target_date_only]
        
        if not matching_rows.empty:
            # Use open price as proxy for 08:00 UTC price
            return float(matching_rows.iloc[0]['open'])
        
        return None
    except Exception as e:
        logger.error(f"Error getting BTC price for {target_date}: {e}")
        return None

def get_historical_option_instruments(currency: str = "BTC", target_expiry_ts_ms: int = None) -> List[Dict]:
    """Get historical option instruments for a specific expiry timestamp"""
    try:
        url = f"{DERIBIT_HISTORY_URL}/public/get_instruments"
        params = {
            'currency': currency,
            'kind': 'option',
            'expired': 'true',
            'include_old': 'true'
        }
        
        data = make_api_request_with_retry(url, params)
        if not data or 'result' not in data:
            return []
        
        all_instruments = data['result']
        logger.info(f"Retrieved {len(all_instruments)} total expired instruments")
        
        if target_expiry_ts_ms:
            # Filter for specific expiry
            matching_instruments = []
            for inst in all_instruments:
                if inst.get('expiration_timestamp') == target_expiry_ts_ms:
                    matching_instruments.append(inst)
            logger.info(f"Found {len(matching_instruments)} instruments for expiry {target_expiry_ts_ms}")
            return matching_instruments
        else:
            return all_instruments
            
    except Exception as e:
        logger.error(f"Error fetching option instruments: {e}")
        return []

def find_closest_2pct_otm_0dte_calls(btc_price: float, instruments: List[Dict], target_date: datetime) -> Optional[Dict]:
    """Find the closest 2% OTM 0DTE call option to the BTC price"""
    target_strike = btc_price * 1.02  # 2% OTM
    
    # Convert target date to timestamp for comparison
    target_timestamp = int(target_date.timestamp() * 1000)
    
    # Filter for call options that expire on the same day (0DTE)
    calls_0dte = []
    for opt in instruments:
        if (opt.get('option_type') == 'call' and 
            opt.get('expiration_timestamp') and 
            opt.get('expiration_timestamp') > target_timestamp):
            # Check if it expires within 24 hours (0DTE)
            expiry_ts = opt['expiration_timestamp']
            hours_to_expiry = (expiry_ts - target_timestamp) / (1000 * 3600)
            if hours_to_expiry <= 24:  # 0DTE (expires within 24 hours)
                calls_0dte.append(opt)
    
    if not calls_0dte:
        logger.debug(f"No 0DTE call options found for {target_date}")
        return None
    
    # Find closest strike to target among 0DTE options
    best_option = None
    min_diff = float('inf')
    
    for option in calls_0dte:
        strike = float(option['strike'])
        diff = abs(strike - target_strike)
        if diff < min_diff:
            min_diff = diff
            best_option = option
    
    logger.debug(f"Found 0DTE option: {best_option.get('instrument_name', 'Unknown')} expiring in {hours_to_expiry:.1f} hours")
    return best_option

def get_historical_trades_24h_window(instrument_name: str, target_date: datetime) -> pd.DataFrame:
    """Fetch historical trades for an instrument in a 24-hour window starting from target date"""
    # Create 00:00 UTC timestamp for the target date (start of day)
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_day = start_of_day.replace(tzinfo=timezone.utc)
    
    # Create 24-hour window (00:00 UTC to 23:59:59 UTC same day)
    start_ts = int(start_of_day.timestamp() * 1000)
    end_ts = int((start_of_day + timedelta(days=1) - timedelta(seconds=1)).timestamp() * 1000)
    
    params = {
        'instrument_name': instrument_name,
        'start_timestamp': start_ts,
        'end_timestamp': end_ts,
        'count': TRADES_COUNT_LIMIT,
        'include_old': 'true'
    }
    
    try:
        url = f"{DERIBIT_HISTORY_URL}/public/get_last_trades_by_instrument_and_time"
        data = make_api_request_with_retry(url, params)
        
        if not data or 'result' not in data or 'trades' not in data['result']:
            logger.debug(f"No trades data for {instrument_name} in 24-hour window on {target_date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()
            
        trades = data['result']['trades']
        if not trades:
            logger.debug(f"No trades found for {instrument_name} in 24-hour window on {target_date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()
            
        df = pd.DataFrame(trades)
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        df = df.drop_duplicates(subset=['trade_id'], keep='first').sort_values(by='timestamp')
        
        logger.debug(f"Found {len(df)} trades for {instrument_name} in 24-hour window on {target_date.strftime('%Y-%m-%d')}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching trades for {instrument_name} in 24-hour window: {e}")
        return pd.DataFrame()

def calculate_option_sale_return(trades_df: pd.DataFrame, btc_shares: float, strike_price: float, next_day_btc_price: float, overwrite_ratio: float = 0.10) -> Dict:
    """Calculate return from selling options based on historical trades"""
    if trades_df.empty:
        return {
            'premium_received': 0.0,
            'option_exercised': False,
            'strike_price': strike_price,
            'expiration_price': next_day_btc_price,
            'instrument_name': '',
            'trades_count': 0,
            'btc_price_at_trade': 0.0
        }
    
    # Get the first trade (our sale price at 08:00 UTC)
    first_trade = trades_df.iloc[0]
    
    # Use the index_price from the trade data (underlying BTC price at time of trade)
    btc_price_at_trade = float(first_trade['index_price'])
    
    # Calculate premium received (we're selling, so we get the bid price)
    # Use the first trade price as our sale price
    premium_per_share_btc = float(first_trade['price'])  # Premium in BTC
    shares_to_sell = btc_shares * overwrite_ratio
    premium_received_btc = premium_per_share_btc * shares_to_sell
    premium_received_usd = premium_received_btc * btc_price_at_trade  # Convert to USD using trade's index price
    
    # Check if option was exercised (expired ITM)
    # Use next day's BTC price at 08:00 UTC to determine exercise
    option_exercised = next_day_btc_price > strike_price if strike_price > 0 else False
    
    return {
        'premium_received': premium_received_usd,  # Return USD amount
        'option_exercised': option_exercised,
        'strike_price': strike_price,
        'expiration_price': next_day_btc_price,
        'instrument_name': first_trade.get('instrument_name', ''),
        'trades_count': len(trades_df),
        'btc_price_at_trade': btc_price_at_trade
    }

def fetch_deribit_data_for_period(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch complete Deribit data for the specified period"""
    logger.info(f"Fetching Deribit data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    results = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            logger.info(f"Processing {current_date.strftime('%Y-%m-%d')}")
            
            # We'll get BTC price from the trades data (index_price), so we don't need to fetch it separately
            # This is more accurate as it's the exact price at the time of the trade
            
            # Get all available option instruments
            all_instruments = get_historical_option_instruments("BTC")
            if not all_instruments:
                logger.warning(f"No option instruments for {current_date}")
                current_date += timedelta(days=1)
                continue
            
            # We need a BTC price to find the 2% OTM strike, so get it from existing data for now
            # The actual BTC price for premium calculation will come from the trades data
            btc_price_for_strike = get_btc_price_from_existing_data(current_date)
            if not btc_price_for_strike:
                logger.warning(f"No BTC price available for strike calculation on {current_date}")
                current_date += timedelta(days=1)
                continue
            
            # Find closest 2% OTM 0DTE call
            target_option = find_closest_2pct_otm_0dte_calls(btc_price_for_strike, all_instruments, current_date)
            if not target_option:
                logger.warning(f"No 2% OTM 0DTE call found for {current_date}")
                current_date += timedelta(days=1)
                continue
            
            logger.info(f"Found 0DTE target option: {target_option.get('instrument_name', 'Unknown')} with strike {target_option.get('strike', 'Unknown')}")
            
            # Get trades in 24-hour window on this date (we sell at first available trade)
            trades_df = get_historical_trades_24h_window(target_option['instrument_name'], current_date)
            
            # Get next day's BTC price at 08:00 UTC (when option expires) - prefer Deribit data
            next_day_btc_price = get_btc_price_from_deribit(current_date + timedelta(days=1))
            if not next_day_btc_price:
                logger.warning(f"No Deribit BTC price for next day {current_date + timedelta(days=1)}, trying fallback...")
                next_day_btc_price = get_btc_price_from_existing_data(current_date + timedelta(days=1))
                if not next_day_btc_price:
                    logger.warning(f"No BTC price for next day {current_date + timedelta(days=1)}")
                    current_date += timedelta(days=1)
                    continue
                else:
                    logger.info(f"Using fallback BTC price for next day: ${next_day_btc_price:,.2f}")
            else:
                logger.info(f"Using Deribit BTC price for next day: ${next_day_btc_price:,.2f}")
            
            # Calculate option sale return
            strike_price = float(target_option.get('strike', 0))
            option_return = calculate_option_sale_return(trades_df, 1.0, strike_price, next_day_btc_price, 0.10)
            
            # Create result record
            result = {
                'date': current_date.date(),
                'btc_price_0800_utc': option_return['btc_price_at_trade'],  # Use BTC price from trades data
                'next_day_btc_price_0800_utc': next_day_btc_price,
                'strike_price': target_option.get('strike', 0),
                'instrument_name': target_option.get('instrument_name', ''),
                'expiration': target_option.get('expiration', ''),
                'premium_received': option_return['premium_received'],
                'option_exercised': option_return['option_exercised'],
                'expiration_price': option_return['expiration_price'],
                'trades_count': option_return['trades_count']
            }
            
            results.append(result)
            exercise_status = "EXERCISED" if option_return['option_exercised'] else "EXPIRED WORTHLESS"
            logger.info(f"Successfully processed {current_date}: 0DTE Strike=${target_option.get('strike', 0):,.0f}, Premium=${option_return['premium_received']:.2f}, {exercise_status}")
            
            # Rate limiting
            time.sleep(REQUEST_DELAY_S)
            
        except Exception as e:
            logger.error(f"Error processing {current_date}: {e}")
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(results)

def main():
    """Main function"""
    logger.info("Starting real Deribit historical data fetch...")
    
    # Define the period to fetch (full PM data range)
    start_date = datetime(2025, 7, 15)
    end_date = datetime(2025, 9, 24)
    
    try:
        # Fetch the data
        deribit_data = fetch_deribit_data_for_period(start_date, end_date)
        
        if not deribit_data.empty:
            # Save to CSV
            output_file = 'data/real_deribit_2pct_otm_data.csv'
            deribit_data.to_csv(output_file, index=False)
            logger.info(f"Saved {len(deribit_data)} records to {output_file}")
            
            # Show summary
            logger.info(f"Data range: {deribit_data['date'].min()} to {deribit_data['date'].max()}")
            logger.info(f"Total premium received: ${deribit_data['premium_received'].sum():.2f}")
            logger.info(f"Options exercised: {deribit_data['option_exercised'].sum()}")
            logger.info(f"Options expired worthless: {(~deribit_data['option_exercised']).sum()}")
        else:
            logger.warning("No data was fetched")
            
    except Exception as e:
        logger.error(f"Error during data fetch: {e}")
        raise

if __name__ == "__main__":
    main()
