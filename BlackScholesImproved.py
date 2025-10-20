import numpy as np

class ExtendedBlackScholes:

    def __init__(self, S0, K, T, dt,
                 v0, kappa, theta, xi, rho,
                 r0, a, b, s,
                 num_simulations=10000):
        """
        Args:
            S0 (float): Initial stock price.
            K (float): Strike price of the option.
            T (float): Time to maturity (in years).
            dt (float): Time step size for the simulation.

            v0 (float): Initial variance of the stock price.
            kappa (float): Rate of mean reversion for variance.
            theta (float): Long-term mean of variance.
            xi (float): Volatility of variance ('vol of vol').
            rho (float): Correlation between stock price and variance processes.

            r0 (float): Initial risk-free interest rate.
            a (float): Rate of mean reversion for interest rate.
            b (float): Long-term mean of interest rate.
            s (float): Volatility of interest rate.

            num_simulations (int): Number of paths to simulate in Monte Carlo.
        """

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


if __name__ == '__main__':
    S0 = 195.71
    sigma = 0.1754
    T = 1/12
    dt = 0.1
    K = 200

    v0 = sigma**2
    kappa = 2.15
    theta = 0.065
    xi = 0.45
    rho = -0.72

    r0 = 0.0423
    a = 0.25
    b = 0.035
    s = 0.015

    extended_model = ExtendedBlackScholes(
        S0=S0, K=K, T=T, dt=dt,
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        r0=r0, a=a, b=b, s=s,
        num_simulations=200000
    )

    call_price = extended_model.price_option(option_type='call')
    put_price = extended_model.price_option(option_type='put')

    print(f"--- Extended Black-Scholes Model ---")
    print(f"European Call Option Price: {call_price:.4f}")
    print(f"European Put Option Price:  {put_price:.4f}")