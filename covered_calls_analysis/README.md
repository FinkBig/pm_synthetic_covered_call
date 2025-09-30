# Covered Calls Analysis

A comprehensive analysis comparing three portfolio strategies using real historical data from Deribit options and Limitless prediction markets.

## Overview

This project analyzes the performance of covered call strategies on Bitcoin using real historical data. It compares three different approaches to generating income from Bitcoin holdings while maintaining exposure to price appreciation.

## Strategies Analyzed

1. **Vanilla BTC**: Simple buy and hold Bitcoin (baseline)
2. **Covered Calls**: Hold BTC + sell 0DTE 2% OTM call options using real Deribit data
3. **PM Covered Call**: Hold BTC + buy "NO" shares on prediction markets (Limitless data)

## Project Structure

```
covered_calls_analysis/
├── main.py                    # Main execution script
├── strategies.py              # Portfolio strategy implementations
├── analysis.py                # Chart generation and metrics calculation
├── config.yaml                # Configuration parameters
├── requirements.txt           # Python dependencies
├── fetch_real_deribit_data.py # Real Deribit data fetcher (optional)
├── data/                      # Data files
│   ├── btc_prices_daily.csv   # Bitcoin price data
│   ├── real_deribit_2pct_otm_data.csv  # Real Deribit 0DTE options data
│   └── limitless_pm.csv       # Limitless prediction market data
└── outputs/                   # Analysis results
    ├── charts/                # Generated charts
    ├── portfolio_metrics.csv  # Performance metrics
    ├── portfolio_metrics.txt  # Formatted metrics table
    └── results.csv            # Raw results data
```

## Key Features

- **Real Historical Data**: Uses actual Deribit 0DTE options trades (82.9% actionable data rate)
- **No Fees**: All strategies assume zero fees for fair comparison
- **Daily Rolling**: Options and PM positions are rolled daily
- **Comprehensive Metrics**: Total return, volatility, Sharpe ratio, max drawdown, win rates

## Data Sources

- **Bitcoin Prices**: Daily OHLCV data
- **Deribit Options**: Real historical 0DTE call options with 2% OTM strikes
- **Limitless PM**: Prediction market data for BTC price levels

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd covered_calls_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main analysis:
   ```bash
   python main.py
   ```

2. View results in the `outputs/` directory:
   - `charts/` - Generated performance charts
   - `portfolio_metrics.csv` - Detailed performance metrics
   - `portfolio_metrics.txt` - Formatted metrics table
   - `results.csv` - Raw results data

## Results Summary

Based on the analysis period (July 15 - September 24, 2025):

| Strategy | Total Return | Annualized Return | Volatility | Max Drawdown | Sharpe Ratio |
|----------|-------------|------------------|------------|--------------|--------------|
| Vanilla BTC | -5.11% | -23.33% | 25.12% | -11.06% | -0.93 |
| Covered Calls | 3.27% | 17.74% | 23.60% | -7.70% | 0.75 |
| PM Covered Call | 49.20% | 660.20% | 51.14% | -10.57% | 12.91 |

**Key Insights:**
- Covered calls strategy outperformed vanilla BTC by 8.38% during the analysis period
- PM covered call replication showed exceptional performance with 49.20% returns
- Covered calls provided better risk-adjusted returns (Sharpe ratio: 0.75 vs -0.93)
- All strategies maintained similar maximum drawdowns around 7-11%

## Configuration

Edit `config.yaml` to adjust:
- Date range for analysis
- Initial capital amount
- OTM percentage for covered calls
- Output settings

## Data Sources

### Real Deribit Data
The analysis uses real historical Deribit 0DTE options data:
- **70 days** of data (July 15 - September 22, 2025)
- **58 actionable days** (82.9% success rate)
- **$561.19 total premium** from real option trades
- **4.3% exercise rate** (realistic for 2% OTM calls)

This provides the most accurate comparison possible using actual historical option trading data.

### Data Fetcher
The `fetch_real_deribit_data.py` script can be used to fetch additional historical data from Deribit's API:
```bash
python fetch_real_deribit_data.py
```

**Note:** This script requires internet access and may take time due to API rate limits.

## Methodology

- **No Fees**: All strategies assume zero fees for fair comparison
- **Daily Rolling**: Options and PM positions are rolled daily
- **Dynamic Allocation**: 10% of portfolio allocated to options/PM positions
- **Real Data**: Uses actual historical trades and market data
- **Comprehensive Metrics**: Total return, volatility, Sharpe ratio, max drawdown, win rates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the analysis
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using external APIs.