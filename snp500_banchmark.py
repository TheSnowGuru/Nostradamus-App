import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def calculate_performance_metrics(portfolio_values, benchmark_values):
    """
    Calculate performance metrics for a portfolio and benchmark.

    Parameters:
    - portfolio_values: Series of portfolio values over time.
    - benchmark_values: Series of benchmark values over time.

    Returns:
    - Dictionary of performance metrics.
    """
    portfolio_returns = portfolio_values.pct_change().dropna()
    benchmark_returns = benchmark_values.pct_change().dropna()

    # Cumulative returns
    portfolio_cum_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    benchmark_cum_return = (1 + benchmark_returns).cumprod().iloc[-1] - 1

    # Annualized returns
    portfolio_annualized = (1 + portfolio_cum_return) ** (1 / (len(portfolio_values) / 252)) - 1
    benchmark_annualized = (1 + benchmark_cum_return) ** (1 / (len(benchmark_values) / 252)) - 1

    # Sharpe Ratio (assuming risk-free rate = 0)
    portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * (252 ** 0.5)
    benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * (252 ** 0.5)

    return {
        "Portfolio Cumulative Return": portfolio_cum_return,
        "Benchmark Cumulative Return": benchmark_cum_return,
        "Portfolio Annualized Return": portfolio_annualized,
        "Benchmark Annualized Return": benchmark_annualized,
        "Portfolio Sharpe Ratio": portfolio_sharpe,
        "Benchmark Sharpe Ratio": benchmark_sharpe
    }

def simulate_equal_weight_strategy(tickers, initial_value, start_date, end_date, rebalance_freq="yearly"):
    """
    Simulate the equal-weighted S&P 500 strategy and compare it with the benchmark.

    Parameters:
    - tickers: List of stock tickers.
    - initial_value: Initial portfolio value.
    - start_date: Backtest start date (YYYY-MM-DD).
    - end_date: Backtest end date (YYYY-MM-DD).
    - rebalance_freq: Rebalancing frequency ("yearly" or "monthly").

    Returns:
    - DataFrame of portfolio and benchmark values.
    """
    # Fetch historical prices
    historical_data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Adj Close"]

    # Fetch S&P 500 (benchmark)
    benchmark_data = yf.download("SPY", start=start_date, end=end_date, progress=False)["Adj Close"]

    portfolio_value = initial_value
    portfolio_values = []
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq="Y" if rebalance_freq == "yearly" else "M")

    for date in rebalance_dates:
        if date > historical_data.index[-1]:
            break

        # Use prices on the rebalancing date
        prices_on_date = historical_data.loc[date]
        num_stocks = len(tickers)

        # Equal weight allocation
        target_value_per_stock = portfolio_value / num_stocks
        shares_to_hold = target_value_per_stock / prices_on_date

        # Rebalance and calculate portfolio value
        portfolio_value = (shares_to_hold * prices_on_date).sum()
        portfolio_values.append({"Date": date, "Portfolio Value": portfolio_value})

    # Prepare portfolio value DataFrame
    portfolio_df = pd.DataFrame(portfolio_values).set_index("Date")

    # Align benchmark and portfolio data
    benchmark_values = benchmark_data.loc[portfolio_df.index]
    portfolio_df["Benchmark Value"] = benchmark_values / benchmark_values.iloc[0] * initial_value

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_df["Portfolio Value"], portfolio_df["Benchmark Value"])
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2%}")

    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df["Portfolio Value"], label="Equal-Weighted Portfolio")
    plt.plot(portfolio_df.index, portfolio_df["Benchmark Value"], label="S&P 500 Benchmark (SPY)")
    plt.title("Portfolio vs Benchmark Performance")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()

    return portfolio_df

# Example usage
if __name__ == "__main__":
    # Example tickers (subset for demonstration)
    sp500_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Replace with full S&P 500 tickers
    initial_portfolio_value = 1000000  # Initial portfolio value
    start_date = "2018-01-01"
    end_date = "2023-12-31"

    # Simulate and compare strategy
    portfolio_results = simulate_equal_weight_strategy(
        sp500_tickers, initial_portfolio_value, start_date, end_date
    )
