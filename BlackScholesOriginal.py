#Black-Scholes model by GregorFabjan on GitHub

import numpy as np
import pandas as pd

def simulate_black_scholes(S0: float, mu: float, sigma: float, T: float, dt: float) -> pd.DataFrame:
    
    N = int(T / dt)  # number of steps
    time = np.linspace(0, T, N + 1)
    
    # Generate random shocks with standard deviation adjusted by sqrt(dt)
    random_shocks = np.random.normal(0, 1, N)
    
    # Calculate the increments in a vectorized manner
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma* np.sqrt(dt)* random_shocks
    
    # Compute the cumulative product for the price path
    price_path = S0 * np.exp(np.insert(np.cumsum(increments), 0, 0))
    
    # Convert to DataFrame with a single column for the simulated path
    return pd.DataFrame(price_path, index=time, columns=['Simulation'])

# Example usage
S0 = 100       # Initial stock price
mu = 0.05      # Expected return
sigma = 0.3    # Volatility
T = 10         # 10 years
dt = 0.5       # 6-month intervals

print(simulate_black_scholes(S0=S0, mu=mu, sigma=sigma, T=T, dt=dt))