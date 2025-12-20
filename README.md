# Pairs Trading Strategy: MNST Correlation Screening + Cointegration Testing

Statistical pairs trading pipeline combining correlation screening with cointegration validation. Developed during an internship at Northrock Capital and significantly updated in December 2025.

## Overview

This project implements a complete pairs trading strategy that:
- **Screens** potential pairs using correlation analysis on sector peers
- **Validates** candidates with Engle-Granger cointegration testing
- **Backtests** the strategy with proper train/test split
- **Evaluates** performance with industry-standard metrics

## Model Specification

The strategy uses a two-stage statistical approach:

### Stage 1: Correlation Screening
Filter candidates by log-price correlation with target stock:
```
corr(log(P_target), log(P_peer)) >= 0.70
```

### Stage 2: Cointegration Testing (Engle-Granger)
For each high-correlation pair, estimate the cointegrating relationship:
```
log(Y) = α + β·log(X) + ε
spread = log(Y) - α - β·log(X)
```

**Cointegration Test:**
- ADF test on residual spread
- H₀: No cointegration (unit root in spread)
- Reject if p-value < 0.05

**Half-Life Estimation:**
```
spread_t = c + φ·spread_{t-1} + e
half_life = -ln(2) / ln(φ)
```

### Trading Signals
Z-score based mean-reversion strategy:
```
z = (spread - μ_rolling) / σ_rolling
```

**Entry/Exit Rules:**
- Long spread when z < -2.0
- Short spread when z > +2.0
- Exit when z crosses 0
- Stop-loss at |z| > 3.5

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_STOCK` | MNST | Stock to find pairs for |
| `MIN_CORRELATION` | 0.70 | Minimum correlation threshold |
| `COINT_PVALUE_THRESHOLD` | 0.05 | Significance level for cointegration |
| `ENTRY_ZSCORE` | 2.0 | Z-score threshold for entry |
| `EXIT_ZSCORE` | 0.0 | Z-score threshold for exit |
| `STOP_LOSS_ZSCORE` | 3.5 | Z-score threshold for stop-loss |
| `LOOKBACK_WINDOW` | 60 | Rolling window for z-score |
| `TRAIN_RATIO` | 0.70 | Train/test split ratio |

## Data

**Source:** Yahoo Finance via `yfinance`

**Universe:** Beverage sector peers
```
MNST, BUD, STZ, SAM, TAP, CELH, PEP, KO, KDP, ABEV, MGPI, WVVI, SBUX
```

**Date Range:** 2022-01-01 to 2024-06-01

**Train/Test Split:**
- Train: First 70% of data (parameter estimation, cointegration testing)
- Test: Remaining 30% (out-of-sample backtest)

## Pipeline

```
1. Data Collection
   └── Fetch adjusted close prices from Yahoo Finance

2. Correlation Screening (Train Data)
   └── Build correlation matrix on log-prices
   └── Filter pairs with correlation >= 0.70

3. Cointegration Testing (Train Data)
   └── OLS regression: log(Y) = α + β·log(X)
   └── ADF test on residual spread
   └── Calculate half-life from AR(1)

4. Pair Selection
   └── Select pair with lowest p-value (if < 0.05)

5. Signal Generation (Test Data)
   └── Compute rolling z-score of spread
   └── Generate entry/exit signals

6. Backtest (Test Data)
   └── Simulate strategy with normalized position weights
   └── Calculate performance metrics
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Total Return (%) | Cumulative return over test period |
| Annualized Return (%) | Geometric annualized return |
| Annualized Volatility (%) | Standard deviation × √252 |
| Sharpe Ratio | Annualized return / volatility |
| Maximum Drawdown (%) | Largest peak-to-trough decline |
| Win Rate (%) | Percentage of profitable trading days |
| Time in Market (%) | Percentage of days with open position |

## Key Concepts

**Why Correlation + Cointegration?**

Correlation measures co-movement but doesn't guarantee mean-reversion. Two stocks can be highly correlated yet drift apart permanently. Cointegration ensures the spread is stationary and will revert to its mean—essential for profitable pairs trading.

**Hedge Ratio (β)**

The OLS coefficient β determines position sizing. For a $1 long position in stock Y, hold $β short in stock X to create a market-neutral spread.

**Half-Life**

Measures how quickly the spread reverts to mean. A half-life of 15 days means deviations decay by 50% every 15 trading days. Shorter half-lives are preferable for active trading.

## Limitations

- Transaction costs not included
- Slippage not modeled
- Assumes continuous liquidity
- Cointegration relationships can break down over time
- Single-pair strategy (no diversification)

## Future Improvements

- Dynamic hedge ratio estimation (Kalman filter)
- Rolling cointegration monitoring with regime detection
- Multi-pair portfolio with correlation constraints
- Transaction cost modeling
- Realistic execution simulation

## Dependencies

```
numpy >= 1.24.0
pandas >= 2.0.0
yfinance >= 0.2.0
statsmodels >= 0.14.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.11.0
```

## Usage

Open `2025_Updated_Notebook.ipynb` in Jupyter and run cells sequentially:

```python
# Configure target and peers
TARGET_STOCK = 'MNST'
PEER_STOCKS = ['BUD', 'STZ', 'SAM', 'TAP', 'CELH', 'PEP', 'KO', 'KDP', 'ABEV', 'MGPI', 'WVVI', 'SBUX']

# Run pipeline - outputs cointegrated pairs and backtest results
```

## Output Files

- `cointegration_screening_results.csv` — All tested pairs with statistics
- `pairs_trading_results.csv` — Daily backtest results for selected pair

## Author

Rush Shah  
December 2025

## License

MIT
