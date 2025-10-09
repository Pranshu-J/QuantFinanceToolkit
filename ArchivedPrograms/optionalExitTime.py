import numpy as np

class LPOptimalExitModel:
    """
    Implements the Longstaff-Schwartz algorithm to find the optimal exit time for a 
    liquidity provider in a constant product AMM (e.g., Uniswap V2).

    This model simulates the external price, arbitrageur trades, and noise trades
    to determine the best time for an LP to withdraw liquidity to maximize their
    returns (fees minus impermanent loss).
    """

    def __init__(self, T, n_steps, n_sims, S0, y0, k, sigma, gamma, lambda_a, lambda_n, r=0.0):
        """
        Initializes the model with all necessary parameters.

        Args:
            T (float): Time horizon in years (e.g., 1 for one year).
            n_steps (int): Number of time steps for the simulation.
            n_sims (int): Number of simulation paths.
            S0 (float): Initial external price of the risky asset.
            y0 (float): Initial reserve of the stable asset (e.g., USDC).
            k (float): The constant product invariant (x * y = k).
            sigma (float): Volatility of the external price.
            gamma (float): Trading fee (e.g., 0.003 for 0.3%).
            lambda_a (float): Arrival rate of arbitrageurs (trades per year).
            lambda_n (float): Arrival rate of noise traders (trades per year).
            r (float): Risk-free interest rate for discounting.
        """
        # --- Store parameters ---
        self.T = T
        self.n_steps = n_steps
        self.n_sims = n_sims
        self.dt = T / n_steps
        self.S0 = S0
        self.y0 = y0
        self.x0 = k / y0
        self.k = k
        self.sigma = sigma
        self.gamma = gamma
        self.lambda_a = lambda_a
        self.lambda_n = lambda_n
        self.r = r
        
        # --- Simulation Results ---
        self.S_paths = np.zeros((n_steps + 1, n_sims))
        self.y_paths = np.zeros((n_steps + 1, n_sims))
        self.x_paths = np.zeros((n_steps + 1, n_sims))
        
        # --- Decision Model ---
        self.continuation_values = []

    def _simulate_paths(self):
        """
        Simulates the paths for the external price (S) and pool reserves (x, y).
        This is the core of the environment simulation, modeling how the pool
        evolves over time due to market movements and trades.
        """
        print("Starting simulation of asset paths and pool dynamics...")
        
        # Initialize paths at time t=0
        self.S_paths[0, :] = self.S0
        self.y_paths[0, :] = self.y0
        self.x_paths[0, :] = self.k / self.y0
        
        # Probabilities of trades occurring in a small time step dt
        p_a = self.lambda_a * self.dt
        p_n = self.lambda_n * self.dt

        for t in range(1, self.n_steps + 1):
            # 1. Evolve external price S using Geometric Brownian Motion
            Z = np.random.standard_normal(self.n_sims)
            self.S_paths[t, :] = self.S_paths[t-1, :] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z
            )

            # Start with previous reserves
            y_t = self.y_paths[t-1, :].copy()
            x_t = self.x_paths[t-1, :].copy()
            
            # 2. Model Arbitrageur Trades
            # Arbitrageurs trade only when the pool price diverges from the external price.
            internal_price = y_t / x_t
            S_t_minus_1 = self.S_paths[t-1, :]
            
            # Condition for arbitrage buy (pool price < external price)
            buy_arb_condition = internal_price < S_t_minus_1
            # Condition for arbitrage sell (pool price > external price)
            sell_arb_condition = internal_price > S_t_minus_1
            
            # Simulate if an arbitrage event happens
            arb_event = np.random.rand(self.n_sims) < p_a
            
            # Execute buy arbitrage
            buy_mask = arb_event & buy_arb_condition
            if np.any(buy_mask):
                # Arbitrageur buys risky asset X, pushing internal price up to S
                # They add dy to the y_pool and remove dx from the x_pool
                dx = x_t[buy_mask] - np.sqrt(self.k / S_t_minus_1[buy_mask])
                dy = (self.k / (x_t[buy_mask] - dx)) - y_t[buy_mask]
                y_t[buy_mask] += dy * (1 - self.gamma) # LP gets fee
                x_t[buy_mask] -= dx

            # Execute sell arbitrage
            sell_mask = arb_event & sell_arb_condition
            if np.any(sell_mask):
                 # Arbitrageur sells risky asset X, pushing internal price down to S
                dy = y_t[sell_mask] - np.sqrt(self.k * S_t_minus_1[sell_mask])
                dx = (self.k / (y_t[sell_mask] - dy)) - x_t[sell_mask]
                x_t[sell_mask] += dx * (1 - self.gamma) # LP gets fee
                y_t[sell_mask] -= dy

            # 3. Model Noise Trader Trades
            # Noise traders trade randomly. For simplicity, we model a small random trade.
            noise_event = np.random.rand(self.n_sims) < p_n
            if np.any(noise_event):
                # 50/50 chance of buying or selling a small amount
                trade_direction = np.random.choice([-1, 1], size=np.sum(noise_event))
                # Trade a small, fixed fraction of the y-reserves
                dy_noise = y_t[noise_event] * 0.01 * trade_direction
                
                dx_noise = (self.k / (y_t[noise_event] + dy_noise)) - x_t[noise_event]
                
                # Update reserves based on trade direction
                y_t[noise_event] += dy_noise * (1 - self.gamma) # Fees are applied
                x_t[noise_event] += dx_noise
            
            # Update paths
            self.y_paths[t, :] = y_t
            self.x_paths[t, :] = x_t
        
        print("Simulation complete.")

    def _get_payoff(self, t):
        """Calculates the LP's payoff if they exit at time step t."""
        # Value of the LP's holdings in the pool
        lp_value = self.x_paths[t, :] * self.S_paths[t, :] + self.y_paths[t, :]
        
        # Value if the LP had just held their initial assets (HODL strategy)
        hodl_value = self.x0 * self.S_paths[t, :] + self.y0
        
        # Payoff is the gain over HODL (fees minus impermanent loss)
        # We take max with 0 because we can't have negative value
        return np.maximum(0, lp_value - hodl_value)

    def solve(self):
        """
        Solves for the optimal exit strategy using backward induction (Longstaff-Schwartz).
        """
        if np.sum(self.S_paths) == 0:
            self._simulate_paths()

        print("Solving for optimal exit strategy using Longstaff-Schwartz...")

        # At the final time step, the value is simply the payoff
        payoff_T = self._get_payoff(self.n_steps)
        cashflow = payoff_T
        
        # Iterate backwards from T-1 to 1
        for t in range(self.n_steps - 1, 0, -1):
            # Discount the future cashflow back one period
            discounted_future_cashflow = cashflow * np.exp(-self.r * self.dt)
            
            # Calculate the immediate payoff if exiting at this time step
            immediate_payoff = self._get_payoff(t)
            
            # Identify paths where immediate payoff is positive (in-the-money paths)
            in_the_money_mask = immediate_payoff > 0
            
            if np.sum(in_the_money_mask) > 10: # Need enough points to run regression
                # State variables for regression: external price S and pool reserve y
                state_variables = np.vstack([
                    self.S_paths[t, in_the_money_mask],
                    self.y_paths[t, in_the_money_mask]
                ]).T
                
                # Y-variable for regression: the discounted future cashflow
                rhs = discounted_future_cashflow[in_the_money_mask]

                # Run a polynomial regression (degree 2) to estimate the continuation value
                # This function E[V(t+1) | S_t, y_t] is the core of the algorithm
                try:
                    # Using polynomial features: 1, S, y, S^2, y^2, Sy
                    X = np.c_[np.ones(state_variables.shape[0]), state_variables, 
                              state_variables[:, 0]**2, state_variables[:, 1]**2, 
                              state_variables[:, 0] * state_variables[:, 1]]
                    
                    beta, _, _, _ = np.linalg.lstsq(X, rhs, rcond=None)
                    
                    # Store the regression coefficients for this time step
                    self.continuation_values.insert(0, beta)
                    
                    # Estimate continuation value for ALL paths using the regression model
                    all_state_vars = np.vstack([self.S_paths[t, :], self.y_paths[t, :]]).T
                    all_X = np.c_[np.ones(all_state_vars.shape[0]), all_state_vars,
                                  all_state_vars[:,0]**2, all_state_vars[:,1]**2,
                                  all_state_vars[:,0] * all_state_vars[:,1]]
                    
                    continuation_value_est = all_X @ beta
                    
                except np.linalg.LinAlgError:
                    # If regression fails, assume continuation value is the mean
                    self.continuation_values.insert(0, None)
                    continuation_value_est = np.mean(rhs)

            else:
                # If not enough data, use the mean or assume it's not worth exercising
                self.continuation_values.insert(0, None)
                continuation_value_est = 0
            
            # --- The Decision Rule ---
            # Decide whether to exercise now or continue
            # If immediate payoff > expected continuation value, we should exercise.
            exercise_mask = immediate_payoff > continuation_value_est
            
            # Update the cashflow for the next iteration
            # If we exercise, the cashflow is the immediate payoff.
            # Otherwise, it's the discounted future cashflow.
            cashflow = np.where(exercise_mask, immediate_payoff, discounted_future_cashflow)

        self.continuation_values.insert(0, None) # No decision at t=0
        print("Solver finished.")
    
    def get_exit_decision(self, t_step, S_current, y_current):
        """
        Given the current state, returns a decision on whether to exit.
        
        Args:
            t_step (int): The current time step index (from 0 to n_steps-1).
            S_current (float): The current external price.
            y_current (float): The current y-reserve in the pool.
            
        Returns:
            (str, float, float): A tuple of (decision, immediate_payoff, continuation_value).
        """
        if not self.continuation_values:
            raise ValueError("Solver has not been run. Call .solve() first.")
        if t_step <= 0 or t_step >= self.n_steps:
             return "HOLD (Edge of Horizon)", 0, 0

        # Calculate immediate payoff
        x_current = self.k / y_current
        lp_value = x_current * S_current + y_current
        hodl_value = self.x0 * S_current + self.y0
        immediate_payoff = max(0, lp_value - hodl_value)
        
        # Get the regression model for this time step
        beta = self.continuation_values[t_step]
        if beta is None:
            return "HOLD (Insufficient Data for Model)", immediate_payoff, 0

        # Estimate continuation value using the stored model
        state_vars = np.array([S_current, y_current])
        X = np.r_[1, state_vars, state_vars[0]**2, state_vars[1]**2, state_vars[0] * state_vars[1]]
        continuation_value = X @ beta
        
        # Decision logic
        if immediate_payoff > continuation_value:
            decision = "EXIT"
        else:
            decision = "HOLD"
            
        return decision, immediate_payoff, continuation_value


# --- Example Usage ---
if __name__ == "__main__":
    # Parameters inspired by the paper's calibration for ETH-USDC on Uniswap V2
    params = {
        'T': 1.0 / 12,        # Time horizon: 1 month
        'n_steps': 30,        # Daily decisions
        'n_sims': 10000,      # Number of simulations for accuracy
        'S0': 2000,           # Initial price of ETH: $2000
        'y0': 200000,         # Initial USDC reserves: $200,000
        'k': 200000 * 100,    # k = y0 * x0 = 200000 * (200000/2000) = 2e10
        'sigma': 0.8,         # Annualized volatility: 80% (typical for crypto)
        'gamma': 0.003,       # 0.3% trading fee
        'lambda_a': 5000,     # Arbitrageur arrival rate (high)
        'lambda_n': 5000,     # Noise trader arrival rate (high)
        'r': 0.02             # Risk-free rate: 2%
    }

    # 1. Initialize and solve the model
    model = LPOptimalExitModel(**params)
    model.solve()
    print("\n" + "="*50)
    print("      Optimal Exit Decision Test      ")
    print("="*50 + "\n")

    # 2. Test the decision function with some example scenarios at day 15 (t_step=15)
    
    # Scenario 1: Price is stable, pool is balanced.
    t_test = 15
    S_test_1 = 2050  # Close to initial price
    y_test_1 = 204000  # Pool still balanced
    decision, payoff, cont_val = model.get_exit_decision(t_test, S_test_1, y_test_1)
    print(f"--- Scenario 1: Stable Market (Day {t_test}) ---")
    print(f"Current State: S=${S_test_1}, y_reserves=${y_test_1:,.0f}")
    print(f"Immediate Payoff (Exit Now): ${payoff:,.2f}")
    print(f"Expected Future Value (Hold): ${cont_val:,.2f}")
    print(f"==> Optimal Decision: {decision}\n")

    # Scenario 2: Price has dropped significantly, causing impermanent loss.
    S_test_2 = 1500  # Significant price drop
    # Pool becomes unbalanced due to arbitrageurs selling ETH for USDC
    y_test_2 = model.k / (model.x0 * np.sqrt(S_test_2/params['S0'])) 
    decision, payoff, cont_val = model.get_exit_decision(t_test, S_test_2, y_test_2)
    print(f"--- Scenario 2: Price Crash (Day {t_test}) ---")
    print(f"Current State: S=${S_test_2}, y_reserves=${y_test_2:,.0f}")
    print(f"Immediate Payoff (Exit Now): ${payoff:,.2f}")
    print(f"Expected Future Value (Hold): ${cont_val:,.2f}")
    print(f"==> Optimal Decision: {decision}\n")

    # Scenario 3: Price has surged, high impermanent loss risk.
    S_test_3 = 2800  # Significant price surge
    y_test_3 = model.k / (model.x0 * np.sqrt(S_test_3/params['S0']))
    decision, payoff, cont_val = model.get_exit_decision(t_test, S_test_3, y_test_3)
    print(f"--- Scenario 3: Price Surge (Day {t_test}) ---")
    print(f"Current State: S=${S_test_3}, y_reserves=${y_test_3:,.0f}")
    print(f"Immediate Payoff (Exit Now): ${payoff:,.2f}")
    print(f"Expected Future Value (Hold): ${cont_val:,.2f}")
    print(f"==> Optimal Decision: {decision}\n")