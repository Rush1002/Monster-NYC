"""
Pairs Trading Strategy: EWA/EWC (Australia/Canada ETFs)
========================================================
This strategy is based on the cointegration relationship between EWA and EWC,
as documented in Ernie Chan's "Algorithmic Trading" (2013).

Both Australia and Canada have economies heavily dependent on natural resources,
leading to a long-term equilibrium relationship between their equity markets.

Author: Rush Shah
Date: December 2024
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKER_1 = 'EWA'  # iShares MSCI Australia ETF
TICKER_2 = 'EWC'  # iShares MSCI Canada ETF

# Data period
START_DATE = '2018-01-01'
END_DATE = '2024-06-01'

# Train/Test split (use first 70% for training, last 30% for testing)
TRAIN_RATIO = 0.70

# Trading parameters
ENTRY_ZSCORE = 2.0      # Enter position when z-score exceeds this
EXIT_ZSCORE = 0.0       # Exit position when z-score crosses zero
STOP_LOSS_ZSCORE = 3.5  # Stop loss if z-score exceeds this

# Lookback for rolling calculations (in trading days)
LOOKBACK_WINDOW = 60

# Initial capital
INITIAL_CAPITAL = 100000

# =============================================================================
# DATA DOWNLOAD AND PREPARATION
# =============================================================================

def download_data(ticker1, ticker2, start, end):
    """Download adjusted close prices for both tickers"""
    print(f"Downloading data for {ticker1} and {ticker2}...")
    print(f"Period: {start} to {end}\n")
    
    data = yf.download([ticker1, ticker2], start=start, end=end, progress=False)['Adj Close']
    data = data.dropna()
    data.columns = [ticker1, ticker2]
    
    print(f"Downloaded {len(data)} trading days of data")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n")
    
    return data

def split_data(data, train_ratio):
    """Split data into training and testing sets"""
    split_idx = int(len(data) * train_ratio)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    print(f"Training set: {len(train)} days ({train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')})")
    print(f"Testing set:  {len(test)} days ({test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')})\n")
    
    return train, test

# =============================================================================
# COINTEGRATION ANALYSIS
# =============================================================================

def test_cointegration(series1, series2, ticker1, ticker2):
    """
    Perform Engle-Granger cointegration test
    Returns: (is_cointegrated, p_value, hedge_ratio)
    """
    print("=" * 60)
    print("COINTEGRATION ANALYSIS (Training Data)")
    print("=" * 60)
    
    # Step 1: Test cointegration
    score, pvalue, _ = coint(series1, series2)
    
    print(f"\nEngle-Granger Cointegration Test:")
    print(f"  Test Statistic: {score:.4f}")
    print(f"  P-Value: {pvalue:.6f}")
    print(f"  Result: {'COINTEGRATED (p < 0.05)' if pvalue < 0.05 else 'NOT COINTEGRATED'}")
    
    # Step 2: Calculate hedge ratio via OLS regression
    # Regress ticker1 on ticker2: ticker1 = alpha + beta * ticker2 + epsilon
    X = add_constant(series2)
    model = OLS(series1, X).fit()
    alpha = model.params.iloc[0]
    hedge_ratio = model.params.iloc[1]
    
    print(f"\nHedge Ratio Estimation (OLS Regression):")
    print(f"  {ticker1} = {alpha:.4f} + {hedge_ratio:.4f} * {ticker2}")
    print(f"  R-squared: {model.rsquared:.4f}")
    
    # Step 3: Calculate spread and test its stationarity
    spread = series1 - hedge_ratio * series2
    
    adf_result = adfuller(spread)
    print(f"\nADF Test on Spread (should be stationary):")
    print(f"  Test Statistic: {adf_result[0]:.4f}")
    print(f"  P-Value: {adf_result[1]:.6f}")
    print(f"  Result: {'STATIONARY (p < 0.05)' if adf_result[1] < 0.05 else 'NOT STATIONARY'}")
    
    # Step 4: Spread statistics
    print(f"\nSpread Statistics:")
    print(f"  Mean: {spread.mean():.4f}")
    print(f"  Std Dev: {spread.std():.4f}")
    print(f"  Half-life: ~{calculate_half_life(spread):.1f} days")
    
    return pvalue < 0.05, pvalue, hedge_ratio, alpha

def calculate_half_life(spread):
    """Calculate mean-reversion half-life using AR(1) model"""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    X = add_constant(spread_lag)
    model = OLS(spread_diff, X).fit()
    
    # Half-life = -log(2) / log(1 + theta)
    theta = model.params.iloc[1]
    if theta >= 0:
        return np.inf
    half_life = -np.log(2) / theta
    return half_life

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(data, hedge_ratio, lookback, entry_z, exit_z, stop_z):
    """
    Generate trading signals based on z-score of the spread
    
    Signal logic:
    - Enter LONG spread (long ticker1, short ticker2) when z-score < -entry_z
    - Enter SHORT spread (short ticker1, long ticker2) when z-score > entry_z
    - Exit when z-score crosses zero
    - Stop loss if z-score exceeds stop_z
    """
    df = data.copy()
    ticker1, ticker2 = df.columns
    
    # Calculate spread
    df['spread'] = df[ticker1] - hedge_ratio * df[ticker2]
    
    # Calculate rolling z-score
    df['spread_mean'] = df['spread'].rolling(window=lookback).mean()
    df['spread_std'] = df['spread'].rolling(window=lookback).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # Initialize position column (1 = long spread, -1 = short spread, 0 = flat)
    df['position'] = 0
    
    position = 0
    positions = []
    
    for i in range(len(df)):
        z = df['zscore'].iloc[i]
        
        if pd.isna(z):
            positions.append(0)
            continue
            
        # Entry logic
        if position == 0:
            if z < -entry_z:
                position = 1  # Long spread (buy ticker1, sell ticker2)
            elif z > entry_z:
                position = -1  # Short spread (sell ticker1, buy ticker2)
        
        # Exit logic
        elif position == 1:  # Currently long spread
            if z >= exit_z or z < -stop_z:
                position = 0
        
        elif position == -1:  # Currently short spread
            if z <= exit_z or z > stop_z:
                position = 0
        
        positions.append(position)
    
    df['position'] = positions
    
    # Calculate position changes for transaction tracking
    df['position_change'] = df['position'].diff().fillna(0)
    
    return df

# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def backtest_strategy(df, hedge_ratio, initial_capital):
    """
    Backtest the pairs trading strategy
    
    Position sizing: Equal dollar amounts on each leg
    - When long spread: Buy $X of ticker1, short $X of ticker2
    - When short spread: Short $X of ticker1, buy $X of ticker2
    """
    ticker1, ticker2 = df.columns[:2]
    
    results = df.copy()
    
    # Calculate daily returns for each stock
    results['ret1'] = results[ticker1].pct_change()
    results['ret2'] = results[ticker2].pct_change()
    
    # Calculate spread return
    # Long spread = long ticker1 + short ticker2
    # Short spread = short ticker1 + long ticker2
    results['spread_return'] = results['position'].shift(1) * (results['ret1'] - hedge_ratio * results['ret2'])
    
    # Calculate strategy returns (position * spread return)
    results['strategy_return'] = results['spread_return'].fillna(0)
    
    # Calculate cumulative returns
    results['cumulative_return'] = (1 + results['strategy_return']).cumprod()
    results['equity'] = initial_capital * results['cumulative_return']
    
    # Track trades
    results['trade'] = (results['position_change'] != 0).astype(int)
    
    return results

def calculate_performance_metrics(results, initial_capital):
    """Calculate comprehensive performance metrics"""
    
    returns = results['strategy_return'].dropna()
    equity = results['equity'].dropna()
    
    # Basic returns
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    
    # Annualized metrics (assuming 252 trading days)
    trading_days = len(returns)
    years = trading_days / 252
    
    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    annualized_volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 0 else 0
    
    # Maximum Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Trade statistics
    positions = results['position']
    trades = results[results['position_change'] != 0]
    num_trades = len(trades) // 2  # Entry + exit = 1 round trip
    
    # Win rate (based on individual return days when in position)
    in_position = returns[positions.shift(1) != 0]
    if len(in_position) > 0:
        win_rate = (in_position > 0).sum() / len(in_position) * 100
    else:
        win_rate = 0
    
    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
    
    # Average trade metrics
    daily_returns_in_position = returns[positions.shift(1) != 0]
    avg_daily_return = daily_returns_in_position.mean() * 100 if len(daily_returns_in_position) > 0 else 0
    
    # Time in market
    time_in_market = (positions != 0).sum() / len(positions) * 100
    
    metrics = {
        'Total Return (%)': total_return,
        'Annualized Return (%)': annualized_return,
        'Annualized Volatility (%)': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown (%)': max_drawdown,
        'Number of Round-Trip Trades': num_trades,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Avg Daily Return When In Position (%)': avg_daily_return,
        'Time in Market (%)': time_in_market,
        'Trading Days': trading_days,
    }
    
    return metrics

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(train_data, test_results, hedge_ratio, ticker1, ticker2, metrics):
    """Create comprehensive visualization of the strategy"""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f'Pairs Trading Strategy: {ticker1}/{ticker2}', fontsize=14, fontweight='bold')
    
    # Plot 1: Price series
    ax1 = axes[0]
    ax1.plot(test_results.index, test_results[ticker1], label=ticker1, alpha=0.8)
    ax1.plot(test_results.index, test_results[ticker2], label=ticker2, alpha=0.8)
    ax1.set_title('Price Series (Test Period)')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Z-Score and trading signals
    ax2 = axes[1]
    ax2.plot(test_results.index, test_results['zscore'], label='Z-Score', color='blue', alpha=0.7)
    ax2.axhline(y=ENTRY_ZSCORE, color='red', linestyle='--', label=f'Entry (±{ENTRY_ZSCORE})')
    ax2.axhline(y=-ENTRY_ZSCORE, color='red', linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=STOP_LOSS_ZSCORE, color='orange', linestyle=':', label=f'Stop Loss (±{STOP_LOSS_ZSCORE})')
    ax2.axhline(y=-STOP_LOSS_ZSCORE, color='orange', linestyle=':')
    ax2.fill_between(test_results.index, 0, test_results['position'], alpha=0.3, 
                     color='green', label='Position')
    ax2.set_title('Z-Score and Trading Signals')
    ax2.set_ylabel('Z-Score / Position')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Equity curve
    ax3 = axes[2]
    ax3.plot(test_results.index, test_results['equity'], label='Strategy Equity', color='green', linewidth=2)
    ax3.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    ax3.set_title(f'Equity Curve (Initial: ${INITIAL_CAPITAL:,})')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown
    ax4 = axes[3]
    rolling_max = test_results['equity'].cummax()
    drawdown = (test_results['equity'] - rolling_max) / rolling_max * 100
    ax4.fill_between(test_results.index, 0, drawdown, color='red', alpha=0.5)
    ax4.set_title('Drawdown')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pairs_trading_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nChart saved as 'pairs_trading_results.png'")

def print_performance_summary(metrics):
    """Print formatted performance summary"""
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE SUMMARY (Out-of-Sample Test Period)")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Ratio' in key or 'Factor' in key:
                print(f"  {key}: {value:.3f}")
            elif '%' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("PAIRS TRADING STRATEGY BACKTEST")
    print(f"Pair: {TICKER_1} / {TICKER_2}")
    print("=" * 60 + "\n")
    
    # Step 1: Download data
    data = download_data(TICKER_1, TICKER_2, START_DATE, END_DATE)
    
    # Step 2: Split into train/test
    train_data, test_data = split_data(data, TRAIN_RATIO)
    
    # Step 3: Test cointegration on training data
    is_cointegrated, pvalue, hedge_ratio, alpha = test_cointegration(
        train_data[TICKER_1], train_data[TICKER_2], TICKER_1, TICKER_2
    )
    
    if not is_cointegrated:
        print("\n⚠️  WARNING: Pair is NOT cointegrated at 5% significance level!")
        print("    Proceeding with backtest anyway, but results may be unreliable.\n")
    else:
        print("\n✓ Pair passes cointegration test. Proceeding with backtest.\n")
    
    # Step 4: Generate signals on test data
    print("=" * 60)
    print("GENERATING TRADING SIGNALS (Test Period)")
    print("=" * 60)
    print(f"  Entry Z-Score: ±{ENTRY_ZSCORE}")
    print(f"  Exit Z-Score: {EXIT_ZSCORE}")
    print(f"  Stop Loss Z-Score: ±{STOP_LOSS_ZSCORE}")
    print(f"  Lookback Window: {LOOKBACK_WINDOW} days")
    
    test_signals = generate_signals(
        test_data, hedge_ratio, LOOKBACK_WINDOW, 
        ENTRY_ZSCORE, EXIT_ZSCORE, STOP_LOSS_ZSCORE
    )
    
    # Step 5: Run backtest
    results = backtest_strategy(test_signals, hedge_ratio, INITIAL_CAPITAL)
    
    # Step 6: Calculate performance metrics
    metrics = calculate_performance_metrics(results, INITIAL_CAPITAL)
    
    # Step 7: Print and visualize results
    print_performance_summary(metrics)
    
    # Step 8: Create visualizations
    print("\nGenerating visualizations...")
    plot_results(train_data, results, hedge_ratio, TICKER_1, TICKER_2, metrics)
    
    # Step 9: Save detailed results to CSV
    results.to_csv('pairs_trading_detailed_results.csv')
    print("Detailed results saved to 'pairs_trading_detailed_results.csv'")
    
    return results, metrics

if __name__ == "__main__":
    results, metrics = main()
