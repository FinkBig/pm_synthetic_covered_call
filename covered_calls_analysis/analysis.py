"""
Simplified analysis and visualization for covered calls strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def generate_charts(btc_data: pd.DataFrame, strategy_returns: Dict[str, pd.DataFrame], output_dir: str):
    """Generate simple, clean charts for each strategy"""
    
    # Set seaborn style for clean charts
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Define colors for each strategy
    strategy_colors = {
        'Vanilla BTC': '#2E86AB',      # Blue
        'Covered Calls': '#A23B72',    # Purple  
        'PM Covered Call': '#F18F01'   # Orange
    }
    
    # Generate individual chart for each strategy
    for strategy_name, returns_df in strategy_returns.items():
        if returns_df.empty:
            continue
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top plot: Portfolio value over time
        color = strategy_colors.get(strategy_name, '#666666')
        ax1.plot(returns_df['timestamp'], returns_df['portfolio_value'], 
                color=color, linewidth=3, alpha=0.8)
        ax1.fill_between(returns_df['timestamp'], returns_df['portfolio_value'], 
                        alpha=0.2, color=color)
        
        ax1.set_title(f'{strategy_name} - Portfolio Value', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Bottom plot: Cumulative returns
        # Use fixed initial capital for consistency with metrics calculation
        initial_value = 10000.0  # All strategies start with $10,000
        cumulative_returns = (returns_df['portfolio_value'] / initial_value - 1) * 100
        
        ax2.plot(returns_df['timestamp'], cumulative_returns, 
                color=color, linewidth=3, alpha=0.8)
        ax2.fill_between(returns_df['timestamp'], cumulative_returns, 0, 
                        alpha=0.2, color=color)
        
        ax2.set_title(f'{strategy_name} - Cumulative Returns', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Cumulative Returns (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add performance metrics as text
        final_return = cumulative_returns.iloc[-1]
        max_value = returns_df['portfolio_value'].max()
        min_value = returns_df['portfolio_value'].min()
        
        metrics_text = f'Total Return: {final_return:.1f}%\nMax Value: ${max_value:,.0f}\nMin Value: ${min_value:,.0f}'
        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        # Save chart with clean filename
        filename = strategy_name.lower().replace(' ', '_') + '.png'
        plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    # Generate Bitcoin price chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Bitcoin price as simple black line with dots
    ax.plot(btc_data['timestamp'], btc_data['close'], 
            color='black', linewidth=2, marker='o', markersize=3, alpha=0.8)
    
    ax.set_title('Bitcoin Price Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Bitcoin Price ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add price metrics
    start_price = btc_data['close'].iloc[0]
    end_price = btc_data['close'].iloc[-1]
    max_price = btc_data['close'].max()
    min_price = btc_data['close'].min()
    price_change = ((end_price - start_price) / start_price) * 100
    
    metrics_text = f'Start: ${start_price:,.0f}\nEnd: ${end_price:,.0f}\nMax: ${max_price:,.0f}\nMin: ${min_price:,.0f}\nChange: {price_change:.1f}%'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bitcoin_price.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Generated individual charts for each strategy and Bitcoin price in {output_dir}")

def save_results(strategy_returns: Dict[str, pd.DataFrame], output_file: str):
    """Save strategy results to CSV with comprehensive metrics"""
    results = []
    
    for strategy_name, returns_df in strategy_returns.items():
        if not returns_df.empty:
            # Calculate comprehensive metrics - use actual initial capital for consistency
            initial_capital = 10000.0  # All strategies start with $10,000
            initial_value = initial_capital
            final_value = returns_df['portfolio_value'].iloc[-1]
            total_return = (final_value / initial_value) - 1
            annualized_return = (1 + total_return) ** (365 / len(returns_df)) - 1
            
            # Daily returns and volatility
            daily_returns = returns_df['portfolio_value'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(365)
            
            # Drawdown analysis
            peak = returns_df['portfolio_value'].cummax()
            drawdown = (returns_df['portfolio_value'] - peak) / peak
            max_drawdown = drawdown.min()
            
            # Risk metrics
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            # sortino_ratio = annualized_return / (daily_returns[daily_returns < 0].std() * np.sqrt(365)) if len(daily_returns[daily_returns < 0]) > 0 else 0
            
            # Additional metrics
            max_value = returns_df['portfolio_value'].max()
            min_value = returns_df['portfolio_value'].min()
            avg_daily_return = daily_returns.mean()
            win_rate = (daily_returns > 0).mean() if len(daily_returns) > 0 else 0
            
            # Calmar ratio (annualized return / max drawdown) - REMOVED
            # calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            results.append({
                'Strategy': strategy_name,
                'Initial_Value': f"${initial_value:,.2f}",
                'Final_Value': f"${final_value:,.2f}",
                'Total_Return': f"{total_return:.2%}",
                'Annualized_Return': f"{annualized_return:.2%}",
                'Volatility': f"{volatility:.2%}",
                'Max_Drawdown': f"{max_drawdown:.2%}",
                'Sharpe_Ratio': f"{sharpe_ratio:.2f}",
                # 'Sortino_Ratio': f"{sortino_ratio:.2f}",
                # 'Calmar_Ratio': f"{calmar_ratio:.2f}",
                'Win_Rate': f"{win_rate:.1%}",
                'Max_Value': f"${max_value:,.2f}",
                'Min_Value': f"${min_value:,.2f}",
                'Avg_Daily_Return': f"{avg_daily_return:.3%}"
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved comprehensive results to {output_file}")
    
    return results_df

def create_metrics_table(strategy_returns: Dict[str, pd.DataFrame], btc_data: pd.DataFrame, output_dir: str):
    """Create a comprehensive metrics table and save as both CSV and formatted text"""
    
    # Calculate metrics for all strategies
    results = []
    
    for strategy_name, returns_df in strategy_returns.items():
        if not returns_df.empty:
            # Basic metrics - use actual initial capital for consistency
            initial_capital = 10000.0  # All strategies start with $10,000
            initial_value = initial_capital
            final_value = returns_df['portfolio_value'].iloc[-1]
            total_return = (final_value / initial_value) - 1
            annualized_return = (1 + total_return) ** (365 / len(returns_df)) - 1
            
            # Risk metrics
            daily_returns = returns_df['portfolio_value'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(365)
            
            peak = returns_df['portfolio_value'].cummax()
            drawdown = (returns_df['portfolio_value'] - peak) / peak
            max_drawdown = drawdown.min()
            
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            # sortino_ratio = annualized_return / (daily_returns[daily_returns < 0].std() * np.sqrt(365)) if len(daily_returns[daily_returns < 0]) > 0 else 0
            # calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Additional metrics
            max_value = returns_df['portfolio_value'].max()
            min_value = returns_df['portfolio_value'].min()
            win_rate = (daily_returns > 0).mean() if len(daily_returns) > 0 else 0
            
            results.append({
                'Strategy': strategy_name,
                'Initial Value': f"${initial_value:,.0f}",
                'Final Value': f"${final_value:,.0f}",
                'Total Return': f"{total_return:.1%}",
                'Annualized Return': f"{annualized_return:.1%}",
                'Volatility': f"{volatility:.1%}",
                'Max Drawdown': f"{max_drawdown:.1%}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                # 'Sortino Ratio': f"{sortino_ratio:.2f}",
                # 'Calmar Ratio': f"{calmar_ratio:.2f}",
                'Win Rate': f"{win_rate:.1%}",
                'Max Value': f"${max_value:,.0f}",
                'Min Value': f"${min_value:,.0f}"
            })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Save as CSV
    csv_file = f'{output_dir}/portfolio_metrics.csv'
    metrics_df.to_csv(csv_file, index=False)
    
    # Create formatted text table
    text_file = f'{output_dir}/portfolio_metrics.txt'
    with open(text_file, 'w') as f:
        f.write("PORTFOLIO PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        # Add analysis period
        start_date = btc_data['timestamp'].min().strftime('%Y-%m-%d')
        end_date = btc_data['timestamp'].max().strftime('%Y-%m-%d')
        f.write(f"Analysis Period: {start_date} to {end_date}\n")
        f.write(f"Total Days: {len(btc_data)}\n\n")
        
        # Write formatted table
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        
        # Add summary insights
        f.write("KEY INSIGHTS:\n")
        f.write("-" * 40 + "\n")
        
        # Find best performers
        best_return = metrics_df.loc[metrics_df['Total Return'].str.rstrip('%').astype(float).idxmax(), 'Strategy']
        best_sharpe = metrics_df.loc[metrics_df['Sharpe Ratio'].str.replace('−', '-').astype(float).idxmax(), 'Strategy']
        lowest_dd = metrics_df.loc[metrics_df['Max Drawdown'].str.rstrip('%').astype(float).idxmax(), 'Strategy']  # Max drawdown is negative, so max = least negative
        
        f.write(f"• Best Total Return: {best_return}\n")
        f.write(f"• Best Risk-Adjusted Return (Sharpe): {best_sharpe}\n")
        f.write(f"• Lowest Drawdown: {lowest_dd}\n")
    
    logger.info(f"Created comprehensive metrics table: {csv_file} and {text_file}")
    return metrics_df
