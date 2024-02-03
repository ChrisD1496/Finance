import yfinance as yf
from pypfopt import black_litterman, risk_models, BlackLittermanModel, DiscreteAllocation, EfficientFrontier, HRPOpt, expected_returns, objective_functions
import riskfolio as rp


class PortfolioConstruction:
    def __init__(self, symbols, start_date, end_date, portfolio_value, risk_free_rate, risk_aversion):
        """
        Initialize the PortfolioConstruction class.

        Args:
            symbols (list): List of stock symbols.
            start_date (str): Start date in the format 'YYYY-MM-DD'.
            end_date (str): End date in the format 'YYYY-MM-DD'.
            portfolio_value (float): Total amount to be invested in the portfolio.
            risk_free_rate (float): Risk-free rate for CAPM.
            risk_aversion (float): Investor's risk aversion factor.
        """
        self.tickers = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.stock_data = self.download_stock_data()

    def download_stock_data(self):
        """
        Download historical stock data for a list of symbols within a specified date range.

        Returns:
            pandas.DataFrame: Historical stock prices.
        """
        stock_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        return stock_data

    def mean_variance(self):
        """
        Perform mean-variance optimization to obtain minimum volatility and maximum Sharpe ratio portfolios.

        Returns:
            dict, dict, dict: Minimum volatility weights, Maximum Sharpe ratio weights, Maximum Quadratic Utility weights.
        """
        S = risk_models.CovarianceShrinkage(self.stock_data, frequency=252).ledoit_wolf()
        returns = expected_returns.capm_return(self.stock_data, risk_free_rate=self.risk_free_rate)

        ef1 = EfficientFrontier(None, S, weight_bounds=(None, None))
        ef1.min_volatility()
        weights_min_vol = ef1.clean_weights()

        ef2 = EfficientFrontier(returns, S)
        ef2.max_sharpe()
        weights_sharp = ef2.clean_weights()

        ef3 = EfficientFrontier(returns, S)
        ef3.max_quadratic_utility(risk_aversion=self.risk_aversion)
        weights_utility = ef3.clean_weights()

        return weights_min_vol, weights_sharp, weights_utility

    def hierarchical_risk_parity_test(self):
        """
        Perform hierarchical risk parity optimization using the riskfolio library.

        Returns:
            dict: Hierarchical risk parity weights.
        """
        return_hist = self.stock_data.pct_change().dropna()
        HRP = rp.HCPortfolio(returns=return_hist, solvers='CLARABEL')
        weights = HRP.optimization(model='HRP', codependence='pearson', rm='MV', rf=self.risk_free_rate, linkage='single', max_k=3, leaf_order=True)
        return weights

    def risk_parity(self):
        """
        Perform risk parity optimization using the riskfolio library.

        Returns:
            dict: Risk parity weights.
        """
        return_hist = self.stock_data.pct_change().dropna()
        RP = rp.Portfolio(returns=return_hist)
        RP.assets_stats(method_mu='hist', method_cov='ledoit')
        weights = RP.rp_optimization(model='Classic', rm='MV', rf=self.risk_free_rate, b=None)
        return weights

    def black_litterman_optimization(self, viewdict, market_caps):
        """
        Perform Black-Litterman optimization.

        Args:
            viewdict (dict): Views on the returns for the next period.
            market_caps (dict): Market capitalizations for each stock.

        Returns:
            dict, dict, float: Portfolio weights, Optimal allocation, Remaining unallocated funds.
        """
        S = risk_models.CovarianceShrinkage(self.stock_data).ledoit_wolf()

        if self.risk_aversion is None:
            delta = black_litterman.market_implied_risk_aversion(self.stock_data, 252, self.risk_free_rate)
        else:
            delta = self.risk_aversion

        if market_caps is None:
            rets = self.stock_data.pct_change().dropna()
            market_prior = rets.mean() * 252
        else:
            market_prior = black_litterman.market_implied_prior_returns(market_caps, delta, S)

        bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict, Q=None, P=None, omega=None,
                                 view_confdience=None, tau=0.05, risk_aversion=delta)

        S_bl = bl.bl_cov()
        ret_bl = bl.bl_returns()

        ef = EfficientFrontier(ret_bl, S_bl)
        ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe()
        weights = ef.clean_weights()

        da = DiscreteAllocation(weights, self.stock_data.iloc[-1], total_portfolio_value=self.portfolio_value)
        alloc, leftover = da.lp_portfolio()

        return weights, alloc, leftover

    def hierarchical_risk_parity(self):
        """
        Perform hierarchical risk parity optimization using the riskfolio library.

        Returns:
            dict, dict, float: Hierarchical risk parity weights, Optimal allocation, Remaining unallocated funds.
        """
        rets = expected_returns.returns_from_prices(self.stock_data)

        hrp = HRPOpt(returns=rets)
        hrp.optimize()
        weights = hrp.clean_weights()

        return weights


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
