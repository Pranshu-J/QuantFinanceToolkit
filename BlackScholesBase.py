# Black-Scholes model and Monte Carlo Option Pricing

import numpy as np
import math

def simulate_stock_end_price(S0: float, r: float, sigma: float, T: float) -> float:
    """
    Simulates the stock's price at the end of the time period T.
    This uses Geometric Brownian Motion under the risk-neutral measure (using 'r' instead of 'mu').
    """
    # Generate a single random shock for the entire period
    Z = np.random.normal(0, 1) 
    
    # Calculate the stock price at time T
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    return ST

def monte_carlo_option_price(S0: float, K: float, T: float, r: float, sigma: float, n_simulations: int, option_type: str) -> float:
    """
    Calculates the option price using Monte Carlo simulation.
    
    Args:
        S0: Initial stock price
        K: Strike price of the option
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility of the underlying stock
        n_simulations: The number of simulations to run
        option_type: 'call' or 'put'
    """
    payoff_sum = 0.0

    for _ in range(n_simulations):
        ST = simulate_stock_end_price(S0, r, sigma, T)

        if option_type == 'call':
            payoff = max(ST - K, 0)
        elif option_type == 'put':
            payoff = max(K - ST, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")
            
        payoff_sum += payoff

    average_payoff = payoff_sum / n_simulations
    
    option_price = average_payoff * math.exp(-r * T)
    
    return option_price

S0 = 195.71          # Initial stock price
sigma = 0.17524       # Volatility
r = 0.0423             # Risk-free interest rate
T = 1/12             # Time to expiration

# Option Specific Parameters
K = 200.00           # Strike Price
n_simulations = 100000 # Number of paths to simulate for higher accuracy

call_price = monte_carlo_option_price(S0=S0, K=K, T=T, r=r, sigma=sigma, n_simulations=n_simulations, option_type='call')
put_price = monte_carlo_option_price(S0=S0, K=K, T=T, r=r, sigma=sigma, n_simulations=n_simulations, option_type='put')

print(f"Number of Simulations: {n_simulations}")
print("-" * 30)
print(f"Call Option Price: ${call_price:.2f}")
print(f"Put Option Price:  ${put_price:.2f}")