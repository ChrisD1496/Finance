# source: https://github.com/adamd1985/quant_research/tree/main

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import yfinance as yf
from statsmodels.tsa.stattools import coint
from itertools import combinations
import seaborn as sns

warnings.filterwarnings("ignore")

class Trading:

    def __init__(self, ticker, start_date, end_date,tickers):
        """
        Initialize the Trading class.

        Parameters:
        - ticker (str): The stock ticker symbol.
        - start_date (str): The start date for data retrieval in 'YYYY-MM-DD' format.
        - end_date (str): The end date for data retrieval in 'YYYY-MM-DD' format.
        """

        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.TS_DAYS_LENGTH = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    def _load_ticker_ts_df(self, ticker):
        """
        Load and cache time series financial data from Yahoo Finance API for a given ticker.

        Parameters:
        - ticker (str): The stock ticker symbol.

        Returns:
        - df (pandas.DataFrame): A DataFrame containing the financial time series data.
        """
        dir_path = './data'
        cached_file_path = f'{dir_path}/{ticker}_{self.start_date}_{self.end_date}.pkl'

        try:
            if os.path.exists(cached_file_path):
                df = pd.read_pickle(cached_file_path)
            else:
                df = yf.download(ticker, start=self.start_date, end=self.end_date)
                os.makedirs(dir_path, exist_ok=True)
                df.to_pickle(cached_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Error loading or caching file with ticker: {ticker}') from e

        return df

    def load_tickers_data(self):
        """
        Load time series financial data for multiple tickers.

        Returns:
        - data_map (dict): A dictionary where keys are stock tickers and values are time series data.
        """
        data_map = {ticker: self._load_ticker_ts_df(ticker) for ticker in self.tickers}
        return data_map

    def calculate_profit(self, signals, prices, transaction_cost=0, risk_free_rate=3):
        """
        Calculate cumulative profit and daily returns based on trading signals and stock prices.

        Parameters:
        - signals (pandas.DataFrame): A DataFrame containing trading signals (1 for buy, -1 for sell).
        - prices (pandas.Series): A Series containing stock prices corresponding to the signal dates.
        - transaction_cost (float): Transaction cost as absolute values per order.
        - risk_free_rate (float): Risk Free Rate as %.

        Returns:
        - daily_returns (pandas.DataFrame): A DataFrame containing daily returns when invested.
        - Sharpe Ratio (float): Contains the Sharpe Ratio of the trading strategy.
        - Max Drawdown Length (int): Contains the Max Drawdown Length of the trading strategy.
        """
        transaction_cost = transaction_cost
        profit = pd.Series(index=prices.index).fillna(0)
        daily_returns = pd.Series(index=prices.index).fillna(0)

        buys = signals[signals['orders'] == 1].index
        sells = signals[signals['orders'] == -1].index
        skip = 0

        for bi in buys:
            if skip > 0:
                skip -= 1
                continue
            sis = sells[sells > bi]

            if len(sis) > 0:
                si = sis[0]

                daily_return = prices[bi:si].pct_change()
                daily_return.iloc[0] = daily_return.iloc[0] - (transaction_cost / prices[bi])
                daily_return.iloc[-1] = daily_return.iloc[-1] - (transaction_cost / prices[si])
                daily_returns.loc[bi:si] = daily_returns.loc[bi:si].add(daily_return.fillna(0), fill_value=0)
                skip = len(buys[(buys > bi) & (buys < si)])
            else:

                daily_return = prices[bi:].pct_change()
                daily_return.iloc[0] = daily_return.iloc[0] - (transaction_cost / prices[bi])
                daily_returns.loc[bi:] = daily_returns.loc[bi:].add(daily_return.fillna(0), fill_value=0)

        cum_daily_profit = ((1 + daily_returns).cumprod() - 1) * 100

        # Calculate Sharpe ratio
        risk_free_rate = risk_free_rate / 100 / 252
        invested_days = daily_returns[daily_returns != 0]  # Exclude days with no investment
        sharpe_ratio = ((invested_days.mean() - risk_free_rate) / invested_days.std()) * np.sqrt(252)

        # Calculate max drawdown
        negative_days = cum_daily_profit[cum_daily_profit < 0].index
        drawdown_lengths = [(group[0], len(group)) for group in pd.Series(negative_days).diff().groupby(negative_days).groups.items()]
        max_drawdown_length = max([length for start_date, length in drawdown_lengths], default=0)

        return cum_daily_profit, sharpe_ratio, max_drawdown_length

    def plot_strategy(self, prices_df, signal_df, profit,title='Trading Strategy'):
        """
        Plot a trading strategy with buy and sell signals and cumulative profit.

        Parameters:
        - prices_df (pandas.Series): A Series containing stock prices.
        - signal_df (pandas.DataFrame): A DataFrame with buy (1) and sell (-1) signals.
        - profit (pandas.Series): A Series containing cumulative profit over time.
        - title (str): The title for the entire figure. Default is 'Trading Strategy'.

        Returns:
        - ax1 (matplotlib.axes.Axes): The top subplot displaying stock prices and signals.
        - ax2 (matplotlib.axes.Axes): The bottom subplot displaying cumulative profit.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': (3, 1)}, figsize=(24, 12))

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price in $')
        ax1.plot(prices_df.index, prices_df, color='g', lw=0.25)

        ax1.plot(signal_df.loc[signal_df.orders == 1.0].index,
                 prices_df[signal_df.orders == 1.0],
                 '^', markersize=12, color='blue', label='Buy')
        ax1.plot(signal_df.loc[signal_df.orders == -1.0].index,
                 prices_df[signal_df.orders == -1.0],
                 'v', markersize=12, color='red', label='Sell')

        ax2.plot(profit.index, profit, color='b')
        ax2.set_ylabel('Cumulative Profit (%)')
        ax2.set_xlabel('Date')
        fig.suptitle(title)

        return ax1, ax2

    def sanitize_data(self,data_map):
        data_sanitized = {}

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        for ticker, data in data_map.items():
            if data is None or len(data) < (self.TS_DAYS_LENGTH / 2):
                # We cannot handle shorter TSs
                continue

            if len(data) > self.TS_DAYS_LENGTH:
                # Normalize to have the same length (self.TS_DAYS_LENGTH)
                data = data[-self.TS_DAYS_LENGTH:]

            # Reindex the time series to match the date range and fill in any blanks (Not Numbers)
            data = data.reindex(date_range)
            data['Adj Close'].replace([np.inf, -np.inf], np.nan, inplace=True)
            data['Adj Close'].interpolate(method='linear', inplace=True)
            data['Adj Close'].fillna(method='pad', inplace=True)
            data['Adj Close'].fillna(method='bfill', inplace=True)

            assert not np.any(np.isnan(data['Adj Close'])) and not np.any(
                np.isinf(data['Adj Close']))

            data_sanitized[ticker] = data

        return data_sanitized

    def find_cointegrated_pairs(self,tickers_ts_map, p_value_threshold=0.2):
        """
        Find cointegrated pairs of stocks based on the Augmented Dickey-Fuller (ADF) test.
        Parameters:
        - tickers_ts_map (dict): A dictionary where keys are stock tickers and values are time series data.
        - p_value_threshold (float): The significance level for cointegration testing.
        Returns:
        - pvalue_matrix (numpy.ndarray): A matrix of cointegration p-values between stock pairs.
        - pairs (list): A list of tuples representing cointegrated stock pairs and their p-values.
        """
        tickers = list(tickers_ts_map.keys())
        n = len(tickers)

        # Extract 'Adj Close' prices into a matrix (each column is a time series)
        adj_close_data = np.column_stack([tickers_ts_map[ticker]['Adj Close'].values for ticker in tickers])
        pvalue_matrix = np.ones((n, n))

        # Calculate cointegration p-values for unique pair combinations
        for i, j in combinations(range(n), 2):
            result = coint(adj_close_data[:, i], adj_close_data[:, j])
            pvalue_matrix[i, j] = result[1]
        pairs = [(tickers[i], tickers[j], pvalue_matrix[i, j])
                 for i, j in zip(*np.where(pvalue_matrix < p_value_threshold))]

        plt.figure(figsize=(26, 26))
        heatmap = sns.heatmap(pvalue_matrix, xticklabels=tickers_ts_map.keys(),
                              yticklabels=tickers_ts_map.keys(), cmap='RdYlGn_r',
                              mask=(pvalue_matrix > (P_VALUE_THRESHOLD)),
                              linecolor='gray', linewidths=0.5)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), size=14)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), size=14)
        plt.show()

        return pvalue_matrix, pairs

    def optimize_simple_moving_average(self,ticker_ts_df, short_window_range, long_window_range):
        """
        Optimize the hyperparameters short_window and long_window for the simple_moving_average function.

        Parameters:
        - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
        - short_window_range (range or list): The range or list of values for short_window.
        - long_window_range (range or list): The range or list of values for long_window.

        Returns:
        - optimal_short_window (int): The optimal value for the short_window hyperparameter.
        - optimal_long_window (int): The optimal value for the long_window hyperparameter.
        """
        best_profit = 0
        optimal_short_window = 0
        optimal_long_window = 0

        # Try different combinations of short_window and long_window
        for short_window in short_window_range:
            for long_window in long_window_range:
                if short_window >= long_window:
                    continue  # Skip invalid combinations

                signal_sma = T.simple_moving_average(ticker_ts_df, short_window=short_window, long_window=long_window)
                profit_series_sma, sharpe_ratio, max_drawdown_length = T.calculate_profit(signal_sma, ticker_ts_df["Adj Close"], transaction_cost=1)
                cumulative_profit = profit_series_sma.iloc[-1]

                if cumulative_profit > best_profit:
                    best_profit = cumulative_profit
                    optimal_short_window = short_window
                    optimal_long_window = long_window

        return optimal_short_window, optimal_long_window

    def simple_moving_average(self, ticker_ts_df, short_window=5, long_window=30):
        """
        Generate trading signals based on a double simple moving average (SMA) strategy.

        Parameters:
        - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
        - short_window (int): The window size for the short-term SMA.
        - long_window (int): The window size for the long-term SMA.

        Returns:
        - signals (pandas.DataFrame): A DataFrame containing the trading signals.
        """
        signals = pd.DataFrame(index=ticker_ts_df.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = ticker_ts_df['Close'].rolling(window=short_window, center=False).mean()
        signals['long_mavg'] = ticker_ts_df['Close'].rolling(window=long_window, center=False).mean()

        signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1, 0)
        signals['orders'] = signals['signal'].diff()
        signals.loc[signals['orders'] == 0, 'orders'] = None
        return signals

    def optimize_naive_momentum(self,ticker_ts_df, nb_conseq_days_range):
        """
        Optimize the hyperparameter nb_conseq_days for the naive_momentum_signals function.

        Parameters:
        - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
        - nb_conseq_days_range (range or list): The range or list of values for nb_conseq_days.

        Returns:
        - optimal_nb_conseq_days (int): The optimal value for the nb_conseq_days hyperparameter.
        """
        best_profit = 0
        optimal_nb_conseq_days = 0


        for nb_days in nb_conseq_days_range:
            signal_momentum = T.naive_momentum_signals(ticker_ts_df, nb_conseq_days=nb_days)
            profit_series_momentum, sharpe_ratio, max_drawdown_length = T.calculate_profit(signal_momentum, ticker_ts_df["Adj Close"], transaction_cost=1)
            cumulative_profit = profit_series_momentum.iloc[-1]

            if cumulative_profit > best_profit:
                best_profit = cumulative_profit
                optimal_nb_conseq_days = nb_days

        return optimal_nb_conseq_days

    def naive_momentum_signals(self, ticker_ts_df, nb_conseq_days=2):
        """
        Generate naive momentum trading signals based on consecutive positive or negative price changes.

        Parameters:
        - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
        - nb_conseq_days (int): The number of consecutive positive or negative days to trigger a signal.

        Returns:
        - signals (pandas.DataFrame): A DataFrame with 'orders' column containing buy (1) and sell (-1) signals.
        """
        signals = pd.DataFrame(index=ticker_ts_df.index)
        signals['orders'] = 0

        price_diff = ticker_ts_df['Adj Close'].diff()

        signal = 0
        cons_day = 0
        reset_flag=0
        first_signal_positiv = False

        for i in range(1, len(ticker_ts_df)):
            if price_diff[i] > 0:
                if reset_flag==-1:
                    cons_day=-1
                reset_flag=1
                cons_day = cons_day + 1 if price_diff[i] > 0 else 0
                if cons_day == nb_conseq_days and signal != 1:
                    signals['orders'].iloc[i] = 1
                    signal = 1
                    first_signal_positiv=True
            elif price_diff[i] < 0:
                if reset_flag==1:
                    cons_day=1
                reset_flag = -1
                cons_day = cons_day - 1 if price_diff[i] < 0 else 0
                if cons_day == -nb_conseq_days and signal != -1 and first_signal_positiv:
                    signals['orders'].iloc[i] = -1
                    signal = -1

        return signals

    def optimize_mean_reversion(self,ticker_ts_df, entry_threshold_range, exit_threshold_range):
        """
        Optimize the hyperparameters entry_threshold and exit_threshold for the mean_reversion function.

        Parameters:
        - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
        - entry_threshold_range (range or list): The range or list of values for entry_threshold.
        - exit_threshold_range (range or list): The range or list of values for exit_threshold.

        Returns:
        - optimal_entry_threshold (float): The optimal value for the entry_threshold hyperparameter.
        - optimal_exit_threshold (float): The optimal value for the exit_threshold hyperparameter.
        """
        best_profit = 0
        optimal_entry_threshold = 0
        optimal_exit_threshold = 0

        # Try different combinations of entry_threshold and exit_threshold
        for entry_threshold in entry_threshold_range:
            for exit_threshold in exit_threshold_range:


                signal_mean_reversion = T.mean_reversion(ticker_ts_df, entry_threshold=entry_threshold,exit_threshold=exit_threshold)
                profit_series_mean_reversion, sharpe_ratio, max_drawdown_length = T.calculate_profit(signal_mean_reversion, ticker_ts_df["Adj Close"], transaction_cost=1)
                cumulative_profit = profit_series_mean_reversion.iloc[-1]

                if cumulative_profit > best_profit:
                    best_profit = cumulative_profit
                    optimal_entry_threshold = entry_threshold
                    optimal_exit_threshold = exit_threshold

        return optimal_entry_threshold, optimal_exit_threshold

    def mean_reversion(self, ticker_ts_df, entry_threshold=1.0, exit_threshold=0.5):
        """
        Generate mean reversion trading signals based on moving averages and thresholds.

        Parameters:
        - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
        - entry_threshold (float): The entry threshold as a multiple of the standard deviation.
        - exit_threshold (float): The exit threshold as a multiple of the standard deviation.

        Returns:
        - signals (pandas.DataFrame): A DataFrame with 'orders' column containing buy (1) and sell (-1) signals.
        """
        signals = pd.DataFrame(index=ticker_ts_df.index)
        signals['mean'] = ticker_ts_df['Adj Close'].rolling(window=20).mean()  # Adjust the window size as needed
        signals['std'] = ticker_ts_df['Adj Close'].rolling(window=20).std()  # Adjust the window size as needed

        # Generate signals based on entry and exit thresholds
        signals['signal'] = 0
        signals.loc[ticker_ts_df['Adj Close'] > (signals['mean'] + entry_threshold * signals['std']), 'signal'] = 1
        signals.loc[ticker_ts_df['Adj Close'] < (signals['mean'] - exit_threshold * signals['std']), 'signal'] = -1

        # Generate 'orders' column by taking the diff of 'signal'
        signals['orders'] = signals['signal'].diff()

        # Set 'orders' to None where there is no change in signal
        signals.loc[signals['orders'] == 0, 'orders'] = None

        return signals

    def optimize_pairs_trading(self, tickers_ts_map, p_value_threshold=0.2):
        """
        Optimize the pairs trading strategy by selecting the pair with the highest return among the three pairs
        with the lowest p-values.

        Parameters:
        - tickers_ts_map (dict): A dictionary where keys are stock tickers and values are time series data.
        - p_value_threshold (float): The significance level for cointegration testing.

        Returns:
        - optimal_pair (tuple): The optimal pair of tickers for pairs trading.
        """
        tickers = list(tickers_ts_map.keys())

        # Find cointegrated pairs with p-values
        _, pairs = self.find_cointegrated_pairs(tickers_ts_map, p_value_threshold=p_value_threshold)

        # Select the three pairs with the lowest p-values
        selected_pairs = sorted(pairs, key=lambda x: x[2])[:5]

        best_profit = 0
        optimal_pair = None

        for pair in selected_pairs:
            ticker_1, ticker_2, _ = pair
            signal_pair_1 = self.pairs_trading_z_score(tickers_ts_map, ticker_1, ticker_2)
            profit_pair_1, sharpe_ratio, max_drawdown_length = self.calculate_profit(signal_pair_1, tickers_ts_map[ticker_1]["Adj Close"])

            signal_pair_2 = self.pairs_trading_z_score(tickers_ts_map, ticker_1, ticker_2, first_ticker=False)
            profit_pair_2, sharpe_ratio, max_drawdown_length = self.calculate_profit(signal_pair_2, tickers_ts_map[ticker_2]["Adj Close"])

            cumulative_profit_pair = profit_pair_1 + profit_pair_2

            if cumulative_profit_pair.iloc[-1] > best_profit:
                best_profit = cumulative_profit_pair.iloc[-1]
                optimal_pair = (ticker_1, ticker_2)

        return optimal_pair
    def pairs_trading_z_score (self,ticker_ts_df,ticker_1, ticker_2, window_size=15,upper_threshold=1,lower_threshold=-1, first_ticker=True):
        """
        Generate trading signals based on z-score analysis of the ratio between two time series.
        Parameters:
        - ticker1_ts (pandas.Series): Time series data for the first security.
        - ticker2_ts (pandas.Series): Time series data for the second security.
        - window_size (int): The window size for calculating z-scores and ratios' statistics.
        - first_ticker (bool): Set to True to use the first ticker as the primary signal source, and False to use the second.

        Returns:
        - signals_df (pandas.DataFrame): A DataFrame with 'signal' and 'orders' columns containing buy (1) and sell (-1) signals.

        """
        ticker1_ts = ticker_ts_df[ticker_1]["Adj Close"]
        ticker2_ts = ticker_ts_df[ticker_2]["Adj Close"]

        ratios = ticker1_ts / ticker2_ts

        ratios_mean = ratios.rolling(
            window=window_size, min_periods=1, center=False).mean()
        ratios_std = ratios.rolling(
            window=window_size, min_periods=1, center=False).std()

        z_scores = (ratios - ratios_mean) / ratios_std

        buy = ratios.copy()
        sell = ratios.copy()

        if first_ticker:
            # These are empty zones, where there should be no signal
            # the rest is signalled by the ratio.
            buy[z_scores > lower_threshold] = 0
            sell[z_scores < upper_threshold] = 0
        else:
            buy[z_scores < upper_threshold] = 0
            sell[z_scores > lower_threshold] = 0

        signals_df = pd.DataFrame(index=ticker1_ts.index)
        signals_df['signal'] = np.where(buy > 0, 1, np.where(sell < 0, -1, 0))
        signals_df['orders'] = signals_df['signal'].diff()
        signals_df.loc[signals_df['orders'] == 0, 'orders'] = None

        return signals_df


if __name__ == '__main__':

    # Parameter single Stock trading strategies

    ticker = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2023-01-01'

    #Hyperparamter to optimize
    short_window_range = range(1, 20)
    long_window_range = range(2, 100)

    nb_conseq_days_range = range(1, 30)

    entry_threshold_range = np.arange(0.0, 3.0, 0.1)  # Example range, adjust as needed
    exit_threshold_range = np.arange(0.0, 3, 0.1)  # Example range, adjust as needed

    # Parameter multiple stocks trading strategie

    bank_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'DB', 'UBS', 'BBVA', 'SAN', 'ING', ' BNPQY', 'HSBC', 'SMFG',
                   'PNC', 'USB', 'BK', 'STT', 'KEY', 'RF', 'HBAN', 'FITB', 'CFG',
                   'BLK', 'ALLY', 'MTB', 'NBHC', 'ZION', 'FFIN', 'FHN', 'UBSI', 'WAL', 'PACW', 'SBCF', 'TCBI', 'BOKF',
                   'PFG', 'GBCI', 'TFC', 'CFR', 'UMBF', 'SPFI', 'FULT', 'ONB', 'INDB', 'IBOC', 'HOMB']

    global_indexes = ['^DJI', '^IXIC', '^GSPC', '^FTSE', '^N225', '^HSI', '^AXJO', '^KS11', '^BFX', '^N100',
                      '^RUT', '^VIX', '^TNX']

    universe_tickers =  bank_stocks + global_indexes
    P_VALUE_THRESHOLD = 0.02

    # Testing the implemented strategies

    T = Trading(ticker, start_date, end_date,universe_tickers)
    ticker_ts_df = T._load_ticker_ts_df(ticker)

    # Using simple moving average strategy

    optimal_short_window, optimal_long_window = T.optimize_simple_moving_average(ticker_ts_df, short_window_range, long_window_range)
    print(f'Optimal short_window: {optimal_short_window}, Optimal long_window: {optimal_long_window}')
    signal_sma = T.simple_moving_average(ticker_ts_df, short_window=optimal_short_window, long_window=optimal_long_window)
    profit_series_sma, sharpe_ratio_sma, max_drawdown_length_sma = T.calculate_profit(signal_sma, ticker_ts_df["Adj Close"], transaction_cost=1)
    print(f'Sharpe Ratio Simple Moving Average Strategy: {sharpe_ratio_sma}, Maximal Drawdown Length Simple Moving Average Strategy: {max_drawdown_length_sma}')

    ax1, _ = T.plot_strategy(ticker_ts_df["Adj Close"], signal_sma, profit_series_sma, title='Simple Moving Average Strategy')
    ax1.plot(signal_sma.index, signal_sma['short_mavg'], linestyle='--', label='Fast SMA')
    ax1.plot(signal_sma.index, signal_sma['long_mavg'], linestyle='--', label='Slow SMA')
    ax1.legend(loc='upper left', fontsize=10)
    plt.show()

    # Using naive momentum strategy

    optimal_nb_conseq_days = T.optimize_naive_momentum(ticker_ts_df, nb_conseq_days_range)
    print(f'Optimal nb_conseq_days: {optimal_nb_conseq_days}')

    signal_momentum = T.naive_momentum_signals(ticker_ts_df,nb_conseq_days=optimal_nb_conseq_days)
    profit_series_momentum, sharpe_ratio_momentum, max_drawdown_length_momentum = T.calculate_profit(signal_momentum, ticker_ts_df["Adj Close"], transaction_cost=1)
    print(f'Sharpe Ratio Momentum Strategy: {sharpe_ratio_momentum}, Maximal Drawdown Length Momentum Strategy: {max_drawdown_length_momentum}')

    ax1, _ = T.plot_strategy(ticker_ts_df["Adj Close"], signal_momentum, profit_series_momentum, title='Momentum Strategy')
    plt.show()

    # Using mean reversion

    optimal_entry_threshold, optimal_exit_threshold = T.optimize_mean_reversion(ticker_ts_df, entry_threshold_range,exit_threshold_range)
    print(f'Optimal entry_threshold: {optimal_entry_threshold}, Optimal exit_threshold: {optimal_exit_threshold}')

    signal_mean_reversion = T.mean_reversion(ticker_ts_df, entry_threshold=optimal_entry_threshold, exit_threshold=optimal_exit_threshold)
    profit_series_mean_reversion, sharpe_ratio_mean_reversion, max_drawdown_length_mean_reversion = T.calculate_profit(signal_mean_reversion, ticker_ts_df["Adj Close"], transaction_cost=1)
    print(f'Sharpe Ratio Mean Reversion Strategy: {sharpe_ratio_mean_reversion}, Maximal Drawdown Length Mean Reversion Strategy: {max_drawdown_length_mean_reversion}')

    ax1, _ = T.plot_strategy(ticker_ts_df["Adj Close"], signal_mean_reversion, profit_series_mean_reversion, title='Mean Reversion Strategy')
    ax1.plot(signal_mean_reversion.index, signal_mean_reversion['mean'], linestyle='--', label="Mean")
    ax1.plot(signal_mean_reversion.index, signal_mean_reversion['mean'] +
             signal_mean_reversion['std'], linestyle='--', label="Ceiling STD")
    ax1.plot(signal_mean_reversion.index, signal_mean_reversion['mean'] -
             signal_mean_reversion['std'], linestyle='--', label="Floor STD")
    ax1.legend(loc='upper left', fontsize=10)
    plt.show()

    #Using pairs trading

    df=T.load_tickers_data()
    clean_df=T.sanitize_data(df)
    optimal_pair = T.optimize_pairs_trading(clean_df, p_value_threshold=P_VALUE_THRESHOLD)
    if optimal_pair:
        ticker_1, ticker_2 = optimal_pair

        # Generate signals and calculate profits for the optimal pair
        signal_pairs_trading_ticker_1 = T.pairs_trading_z_score(clean_df, ticker_1, ticker_2)
        signal_pairs_trading_ticker_2 = T.pairs_trading_z_score(clean_df, ticker_1, ticker_2, first_ticker=False)

        profit_pairs_trading_ticker_1, sharpe_ratio, max_drawdown_length = T.calculate_profit(signal_pairs_trading_ticker_1,
                                                           clean_df[ticker_1]["Adj Close"])
        profit_pairs_trading_ticker_2, sharpe_ratio, max_drawdown_length = T.calculate_profit(signal_pairs_trading_ticker_2,
                                                           clean_df[ticker_2]["Adj Close"])

        # Plot individual strategies for the optimal pair
        plt.figure(figsize=(26, 18))
        ax1, _ = T.plot_strategy(clean_df[ticker_1]["Adj Close"], signal_pairs_trading_ticker_1,
                                 profit_pairs_trading_ticker_1, title='Pairs Trading Stock 1')
        ax2, _ = T.plot_strategy(clean_df[ticker_2]["Adj Close"], signal_pairs_trading_ticker_2,
                                 profit_pairs_trading_ticker_2, title='Pairs Trading Stock 2')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_title(f'{ticker_2} Paired with {ticker_1}', fontsize=18)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.set_title(f'{ticker_1} Paired with {ticker_2}', fontsize=18)
        plt.tight_layout()
        plt.show()

        # Plot combined cumulative profit for the optimal pair
        plt.figure(figsize=(12, 6))
        cumulative_profit_combined = profit_pairs_trading_ticker_1 + profit_pairs_trading_ticker_2
        ax2_combined = cumulative_profit_combined.plot(label='Profit%', color='green')
        plt.legend(loc='upper left', fontsize=10)
        plt.title(f'{ticker_1} & {ticker_2} Paired - Cumulative Profit', fontsize=18)
        plt.tight_layout()
        plt.show()
