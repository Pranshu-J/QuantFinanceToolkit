import numpy as np
import pandas as pd

class ExtendedBlackScholes:

    def __init__(self, S0, K, T, dt,
                 v0, kappa, theta, xi, rho,
                 r0, a, b, s,
                 num_simulations=10000):
       

        self.S0 = S0
        self.K = K
        self.T = T
        self.dt = dt
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r0 = r0
        self.a = a
        self.b = b
        self.s = s
        self.num_simulations = num_simulations
        self.num_steps = int(T / dt)

    def _generate_correlated_paths(self):

        Z_v = np.random.normal(0.0, 1.0, (self.num_simulations, self.num_steps))
        Z_S_ind = np.random.normal(0.0, 1.0, (self.num_simulations, self.num_steps))
        Z_r = np.random.normal(0.0, 1.0, (self.num_simulations, self.num_steps))

        Z_S = self.rho * Z_v + np.sqrt(1.0 - self.rho**2) * Z_S_ind

        return Z_S, Z_v, Z_r

    def price_option(self, option_type='call'):

        Z_S, Z_v, Z_r = self._generate_correlated_paths()

        S = np.full((self.num_simulations, self.num_steps + 1), self.S0)
        v = np.full((self.num_simulations, self.num_steps + 1), self.v0)
        r = np.full((self.num_simulations, self.num_steps + 1), self.r0)

        for t in range(self.num_steps):
            v[:, t] = np.maximum(v[:, t], 0)

            S[:, t+1] = S[:, t] * np.exp((r[:, t] - 0.5 * v[:, t]) * self.dt +
                                         np.sqrt(v[:, t] * self.dt) * Z_S[:, t])

            v[:, t+1] = (v[:, t] + self.kappa * (self.theta - v[:, t]) * self.dt +
                         self.xi * np.sqrt(v[:, t] * self.dt) * Z_v[:, t])

            r[:, t+1] = (r[:, t] + self.a * (self.b - r[:, t]) * self.dt +
                         self.s * np.sqrt(self.dt) * Z_r[:, t])

        avg_interest_rate = np.mean(r, axis=1)

        if option_type == 'call':
            payoff = np.maximum(S[:, -1] - self.K, 0)
        elif option_type == 'put':
            payoff = np.maximum(self.K - S[:, -1], 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

        discounted_payoff = payoff * np.exp(-avg_interest_rate * self.T)

        option_price = np.mean(discounted_payoff)

        return option_price

# Fixed parameters
dt = 0.1
kappa = 2.15
theta = 0.065
xi = 0.45
rho = -0.72
r0 = 0.0423
a = 0.25
b = 0.035
s = 0.015
num_simulations = 200000

# Read the CSV file
df = pd.read_csv('predicted_call_prices.csv')
df = df.head(200)  # Limit to first 20 rows

# Debug: Print columns to verify
print("CSV Columns:", df.columns.tolist())
print(df.head())

# Prepare output DataFrame
results = []

for index, row in df.iterrows():
    contract_symbol = row['contractSymbol']
    K = row['strike']
    sigma = row['impliedVolatility']
    S0 = row['price']
    remaining = row['remaining']
    T = remaining / 365.0  # Convert days to years
    v0 = sigma ** 2
    
    extended_model = ExtendedBlackScholes(
        S0=S0, K=K, T=T, dt=dt,
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        r0=r0, a=a, b=b, s=s,
        num_simulations=num_simulations
    )
    
    # Calculate call option price
    call_price = extended_model.price_option(option_type='call')
    
    # Append to results
    results.append({'contractSymbol': contract_symbol, 'predicted_call_price': call_price})

# Create output DataFrame and save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv('extended_bs_call_prices.csv', index=False)

print("Processing complete. Results saved to 'extended_bs_call_prices.csv'")
print(output_df)