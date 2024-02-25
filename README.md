# Financial Modeling Repository

Explore solutions to diverse financial problems with this Python repository. Find practical code snippets and notebooks for informed decision-making.

## Portfolio Construction (Chapter 1)

The `PortfolioConstruction` class provides a comprehensive set of tools for effective portfolio construction, covering various optimization techniques and risk management strategies. Whether you are a novice or an experienced investor, these Python scripts empower you to make informed decisions in managing your investment portfolio.

### Class Overview

#### Initialization

- **symbols:** List of stock symbols.
- **start_date:** Start date in 'YYYY-MM-DD' format.
- **end_date:** End date in 'YYYY-MM-DD' format.
- **portfolio_value:** Total investment portfolio value.
- **risk_free_rate:** Risk-free rate for CAPM.
- **risk_aversion:** Investor's risk aversion factor.

### Functions

1. **Download Stock Data:**
   - Description: Downloads historical stock data within the specified date range.
   - Usage:
     ```python
     stock_data = portfolio.download_stock_data()
     ```

2. **Mean-Variance Optimization:**
   - Description: Implements mean-variance optimization to obtain minimum volatility, maximum Sharpe ratio, and maximum quadratic utility portfolios.
   - Usage:
     ```python
     weights_min_vol, weights_sharp, weights_utility = portfolio.mean_variance()
     ```

3. **Hierarchical Risk Parity Test:**
   - Description: Performs hierarchical risk parity optimization using the riskfolio library.
   - Usage:
     ```python
     weights_hrp_test = portfolio.hierarchical_risk_parity_test()
     ```

4. **Risk Parity Optimization:**
   - Description: Utilizes the riskfolio library to achieve risk parity in the portfolio.
   - Usage:
     ```python
     weights_rp = portfolio.risk_parity()
     ```

5. **Black-Litterman Optimization:**
   - Description: Implements the Black-Litterman model, incorporating subjective views on expected returns.
   - Usage:
     ```python
     weights_bl, allocation_bl, leftover_bl = portfolio.black_litterman_optimization(viewdict, market_caps)
     ```

### Usage Example

```python
# Define stock symbols and date range
symbols = ["AAPL", "MSFT", "META", "AMZN", "XOM", "UNH", "JNJ", "V"]
start_date = "2023-01-01"
end_date = "2024-01-01"
portfolio_value = 30000
risk_free_rate = 0.02
risk_aversion = 2

# Create an instance of the PortfolioConstruction class
portfolio = PortfolioConstruction(symbols, start_date, end_date, portfolio_value, risk_free_rate, risk_aversion)

# Define views and market caps
viewdict = {
    'AAPL': 0.10,
    'MSFT': 0.13,
    'META': 0.08,
    'AMZN': -0.01,
    'XOM': 0.03,
    'UNH': 0.21,
    'JNJ': -0.2,
    'V': 0.05
}
market_caps = {t: yf.Ticker(t).info["marketCap"] for t in symbols}

# Print results for each optimization method
print("\nHierarchical Risk Parity Test Weights:")
print(portfolio.hierarchical_risk_parity_test())

# Perform hierarchical risk parity optimization and print results
print("\nHierarchical Risk Parity Weights:")
weights_hrp = portfolio.hierarchical_risk_parity()
print(weights_hrp)

print("\nRisk Parity Weights:")
print(portfolio.risk_parity())

weights_min_vol, weights_sharp, weights_utility = portfolio.mean_variance()
print("\nMinimum Volatility Weights:")
print(weights_min_vol)
print("\nMaximum Sharpe Ratio Weights:")
print(weights_sharp)
print("\nMaximum Quadratic Utility Weights:")
print(weights_utility)

# Perform Black-Litterman optimization and print results
print("\nBlack-Litterman Weights:")
weights_bl, allocation_bl, leftover_bl = portfolio.black_litterman_optimization(viewdict, market_caps)
print(weights_bl)
print(allocation_bl)
print(f"Remaining Unallocated Funds: ${leftover_bl:.2f}")
```
Explore and customize the provided functions based on your portfolio construction needs. The class allows for easy extension to incorporate additional optimization techniques in future chapters.

### Citation


```bibtex
@misc{riskfolio,
    author = {Dany Cajas},
    title = {Riskfolio-Lib (5.0.0)},
    year = {2024},
    url = {https://github.com/dcajasn/Riskfolio-Lib},
}

@article{Martin2021,
  doi = {10.21105/joss.03066},
  url = {https://doi.org/10.21105/joss.03066},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {61},
  pages = {3066},
  author = {Robert Andrew Martin},
  title = {PyPortfolioOpt: portfolio optimization in Python},
  journal = {Journal of Open Source Software}
}
```

## Simple Trading Strategies (Chapter 2)

Explore and implement simple yet effective trading strategies using Python. This chapter provides practical code snippets and an object-oriented structure for four trading strategies: Moving Average, Momentum, Mean Reversion, and Pairs Trading.

### Moving Average Strategy

#### Class Overview

##### Initialization

- **tickers:** List of stock tickers for analysis.
- **start_date:** Start date for data retrieval in 'YYYY-MM-DD' format.
- **end_date:** End date for data retrieval in 'YYYY-MM-DD' format.

##### Functions

1. **Load Tickers Data:**
   - Description: Loads time series financial data for multiple tickers.
   - Usage:
     ```python
     data_map = trading.load_tickers_data()
     ```

2. **Calculate Profit:**
   - Description: Calculates cumulative profit based on trading signals and stock prices.
   - Usage:
     ```python
     cumulative_profit = trading.calculate_profit(signals, prices)
     ```

3. **Plot Strategy:**
   - Description: Plots a trading strategy with buy and sell signals and cumulative profit.
   - Usage:
     ```python
     ax1, ax2 = trading.plot_strategy(prices_df, signal_df, profit)
     ```

#### Optimization Process

Optimize the strategy by testing various combinations of parameters, such as transaction costs and window sizes. Select optimal parameters based on cumulative returns over historical stock data.

### Momentum Strategy

#### Class Overview

##### Initialization

- **tickers:** List of stock tickers for analysis.
- **start_date:** Start date for data retrieval in 'YYYY-MM-DD' format.
- **end_date:** End date for data retrieval in 'YYYY-MM-DD' format.
- **nb_conseq_days_range:** Range of values for the number of consecutive days.

##### Functions

1. **Load Tickers Data:**
   - Description: Loads time series financial data for multiple tickers.
   - Usage:
     ```python
     data_map = trading.load_tickers_data()
     ```

2. **Optimize Naive Momentum:**
   - Description: Optimizes the hyperparameter nb_conseq_days for the naive momentum strategy.
   - Usage:
     ```python
     optimal_nb_conseq_days = trading.optimize_naive_momentum(ticker_ts_df, nb_conseq_days_range)
     ```

3. **Naive Momentum Signals:**
   - Description: Generates naive momentum trading signals based on consecutive positive or negative price changes.
   - Usage:
     ```python
     signals_momentum = trading.naive_momentum_signals(ticker_ts_df, nb_conseq_days=optimal_nb_conseq_days)
     ```

#### Optimization Process

Optimize the strategy by determining the optimal number of consecutive days. Maximize cumulative returns over the specified date range.

### Mean Reversion Strategy

#### Class Overview

##### Initialization

- **tickers:** List of stock tickers for analysis.
- **start_date:** Start date for data retrieval in 'YYYY-MM-DD' format.
- **end_date:** End date for data retrieval in 'YYYY-MM-DD' format.
- **entry_threshold_range:** Range of values for the entry threshold.
- **exit_threshold_range:** Range of values for the exit threshold.

##### Functions

1. **Load Tickers Data:**
   - Description: Loads time series financial data for multiple tickers.
   - Usage:
     ```python
     data_map = trading.load_tickers_data()
     ```

2. **Optimize Mean Reversion:**
   - Description: Optimizes the hyperparameters entry_threshold and exit_threshold for the mean reversion strategy.
   - Usage:
     ```python
     optimal_entry_threshold, optimal_exit_threshold = trading.optimize_mean_reversion(ticker_ts_df, entry_threshold_range, exit_threshold_range)
     ```

3. **Mean Reversion Signals:**
   - Description: Generates mean reversion trading signals based on moving averages and thresholds.
   - Usage:
     ```python
     signals_mean_reversion = trading.mean_reversion(ticker_ts_df, entry_threshold=optimal_entry_threshold, exit_threshold=optimal_exit_threshold)
     ```

#### Optimization Process

Optimize the strategy by experimenting with different entry and exit threshold values. Select parameters that maximize cumulative returns, considering transaction costs.

### Pairs Trading Strategy

#### Class Overview

##### Initialization

- **tickers:** List of stock tickers for analysis.
- **start_date:** Start date for data retrieval in 'YYYY-MM-DD' format.
- **end_date:** End date for data retrieval in 'YYYY-MM-DD' format.
- **p_value_threshold:** The significance level for cointegration testing.

##### Functions

1. **Load Tickers Data:**
   - Description: Loads time series financial data for multiple tickers.
   - Usage:
     ```python
     data_map = trading.load_tickers_data()
     ```

2. **Find Cointegrated Pairs:**
   - Description: Finds cointegrated pairs of stocks based on the Augmented Dickey-Fuller test.
   - Usage:
     ```python
     cointegrated_pairs = trading.find_cointegrated_pairs(ticker_ts_df, p_value_threshold)
     ```

3. **Optimize Pairs Trading:**
   - Description: Optimizes the pair for the pairs trading strategy based on cumulative returns.
   - Usage:
     ```python
     optimal_pair = trading.optimize_pairs_trading(ticker_ts_df, cointegrated_pairs)
     ```

4. **Pairs Trading Z-Score:**
   - Description: Generates pairs trading signals based on z-score analysis of the price ratio.
   - Usage:
     ```python
     signals_pairs_trading = trading.pairs_trading_z_score(ticker_ts_df, pair=optimal_pair)
     ```

#### Optimization Process

Optimize the strategy by finding cointegrated pairs with low p-values. Select the optimal pair based on cumulative returns, considering transaction costs.

### Conclusion and Citation

This chapter introduces and optimizes four simple trading strategies. The code is structured in an object-oriented manner, allowing for easy customization and extension. The strategies are optimized for specific parameters to maximize cumulative returns over historical data. Feel free to explore, modify, and integrate these strategies into your trading projects.

For the original code and further details, refer to the GitHub repository of Adam Darmanin: [https://github.com/adamd1985/quant_research](https://github.com/adamd1985/quant_research).
