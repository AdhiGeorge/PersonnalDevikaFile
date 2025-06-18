import numpy as np
import pandas as pd


def calculate_vix(option_data: pd.DataFrame, risk_free_rate: float, T: float) -> float:
    """
    Calculate the VIX score based on option prices and implied volatilities.

    Parameters:
    - option_data: DataFrame containing columns ['strike_price', 'bid_price', 'ask_price', 'type'].
                   'type' should be 'call' or 'put'.
    - risk_free_rate: Risk-free interest rate as a decimal (e.g., 0.05 for 5%).
    - T: Time to expiration in years (e.g., 30 days = 30/365).

    Returns:
    - VIX score as a float (percentage).
    """
    # Input validation
    required_columns = {'strike_price', 'bid_price', 'ask_price', 'type'}
    if not required_columns.issubset(option_data.columns):
        raise ValueError(
            f"option_data must contain the following columns: {required_columns}"
        )
    if not isinstance(risk_free_rate, (float, int)) or risk_free_rate < 0:
        raise ValueError("risk_free_rate must be a non-negative float or integer.")
    if not isinstance(T, (float, int)) or T <= 0:
        raise ValueError("T must be a positive float or integer representing time to expiration in years.")
    if not option_data['type'].isin(['call', 'put']).all():
        raise ValueError("The 'type' column must only contain 'call' or 'put' values.")

    try:
        # Calculate mid prices for options
        option_data['mid_price'] = (option_data['bid_price'] + option_data['ask_price']) / 2

        # Separate call and put options
        calls = option_data[option_data['type'] == 'call']
        puts = option_data[option_data['type'] == 'put']

        # Combine call and put options by strike price
        combined = pd.merge(calls, puts, on='strike_price', suffixes=('_call', '_put'))

        # Calculate the contribution of each strike price to the variance
        combined['price_contribution'] = (
            (combined['mid_price_call'] + combined['mid_price_put']) / 2
        ) / combined['strike_price'] ** 2

        # Sum the contributions and multiply by scaling factors
        variance = (2 / T) * combined['price_contribution'].sum()

        # Adjust for scaling with risk-free rate
        variance /= np.exp(risk_free_rate * T)

        # Convert variance to volatility (standard deviation)
        volatility = np.sqrt(variance)

        return volatility * 100  # Return as a percentage (VIX score)

    except KeyError as e:
        raise KeyError(f"Missing required column in option_data: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during VIX calculation: {e}")


# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'strike_price': [100, 110, 120, 130],
        'bid_price': [2.5, 2.0, 1.8, 1.5],
        'ask_price': [3.0, 2.5, 2.2, 1.8],
        'type': ['call', 'call', 'put', 'put']
    }

    option_data = pd.DataFrame(data)
    risk_free_rate = 0.05  # 5% annual interest rate
    T = 30 / 365  # 30 days to expiration

    try:
        vix_score = calculate_vix(option_data, risk_free_rate, T)
        print(f"Calculated VIX score: {vix_score:.2f}")
    except Exception as e:
        print(f"Failed to calculate VIX score: {e}")