# Financial Modeling Repository

Explore solutions to diverse financial problems with this Python repository. Find practical code snippets and notebooks for informed decision-making.

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

## Citation


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