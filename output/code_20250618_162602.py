import numpy as np
import pandas as pd

def calculate_vix(historical_prices: pd.Series) -> float:
    """
    Calculate a simplified VIX-like score based on historical price data.

    Args:
        historical_prices (pd.Series): A Pandas Series containing historical price data.

    Returns:
        float: A simplified VIX score representing the annualized volatility.

    Raises:
        ValueError: If the input data is invalid (e.g., contains NaN or insufficient data).
    """
    # Validate input
    if not isinstance(historical_prices, pd.Series):
        raise TypeError("Input data must be a Pandas Series containing historical prices.")
    if historical_prices.isnull().any():
        raise ValueError("Historical price data contains missing values.")
    if len(historical_prices) < 2:
        raise ValueError("Insufficient data to calculate VIX. At least 2 data points are required.")

    # Calculate daily log returns
    log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()

    # Validate that log_returns is not empty (edge case: constant price series)
    if log_returns.empty:
        raise ValueError("Log returns calculation resulted in an empty series. Check the input data.")

    # Calculate standard deviation of daily log returns
    std_dev = log_returns.std()

    # Annualize the volatility (assuming 252 trading days per year)
    annualized_volatility = std_dev * np.sqrt(252)

    return annualized_volatility

if __name__ == "__main__":
    # Example usage
    try:
        # Generate sample historical price data (random walk simulation)
        np.random.seed(42)
        days = 100  # Number of days of simulated price data
        simulated_prices = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, days)) * 100)

        # Calculate the VIX score
        vix_score = calculate_vix(simulated_prices)
        print(f"Calculated VIX score (simplified): {vix_score:.2f}")
    except Exception as e:
        print(f"An error occurred: {e}")