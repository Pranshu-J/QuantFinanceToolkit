import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

class QHRModel:
    def __init__(self, Lambda, b, alpha, beta, Gamma, x0, y0):
        """
        Initializes the QHR model with its parameters.
        
        Parameters:
        Lambda (np.array): p x p matrix.
        b (np.array): p-dimensional column vector.
        alpha (float): Scalar > 0.
        beta (np.array): p-dimensional column vector.
        Gamma (np.array): p x p symmetric matrix.
        x0 (float): Initial log-price.
        y0 (np.array): Initial p-dimensional offset vector.
        """
        self.Lambda = Lambda
        self.b = b.reshape(-1, 1) # Ensure b is a column vector
        self.alpha = alpha
        self.beta = beta.reshape(-1, 1) # Ensure beta is a column vector
        self.Gamma = Gamma
        self.x0 = x0
        self.y0 = y0.reshape(-1, 1) # Ensure y0 is a column vector
        
        self.p = self.Lambda.shape[0] # Dimension of the offset process

    def simulate_paths(self, T, N, M):
        """
        Simulates the QHR model using the Euler-Maruyama scheme.

        Parameters:
        T (float): Time horizon (e.g., 1 year).
        N (int): Number of time steps.
        M (int): Number of simulation paths.

        Returns:
        x_paths (np.array): (M, N+1) array of log-price paths.
        y_paths (np.array): (M, p, N+1) array of offset vector paths.
        """
        dt = T / N

        # Initialize arrays to store paths
        x_paths = np.zeros((M, N + 1))
        y_paths = np.zeros((M, self.p, N + 1))

        # Set initial conditions for all paths
        x_paths[:, 0] = self.x0
        y_paths[:, :, 0] = self.y0.flatten()

        # Generate all random shocks at once for efficiency
        Z = np.random.standard_normal((M, N))
        dW = np.sqrt(dt) * Z

        # Simulation loop
        for i in range(N):
            y_current = y_paths[:, :, i] # Shape (M, p)

            # 1. Calculate variance for each path [cite: 112]
            # y_current shape is (M, p), need to reshape for matrix math
            y_current_col = y_current.reshape(M, self.p, 1)
            
            # Quadratic term: y_t^T * Gamma * y_t
            quad_term = np.einsum('mpi,pq,mqi->m', y_current_col, self.Gamma, y_current_col)
            # Linear term: 2 * beta^T * y_t
            linear_term = 2 * (self.beta.T @ y_current_col).flatten()
            
            sigma2 = self.alpha + linear_term + quad_term
            sigma2 = np.maximum(sigma2, 1e-9) # Ensure variance is non-negative
            sigma = np.sqrt(sigma2)

            # 2. Update the offset vector y_t [cite: 111]
            # Reshape sigma and dW for broadcasting
            sigma_col = sigma.reshape(M, 1)
            dW_col = dW[:, i].reshape(M, 1)

            # dy = -Lambda*y*dt + b*sigma*dW
            y_drift = - (self.Lambda @ y_current.T).T * dt
            y_diffusion = self.b.T * sigma_col * dW_col
            y_paths[:, :, i+1] = y_current + y_drift + y_diffusion

            # 3. Update the log-price x_t [cite: 111]
            # dx = -0.5*sigma^2*dt + sigma*dW
            x_paths[:, i+1] = x_paths[:, i] - 0.5 * sigma2 * dt + sigma * dW[:, i]

        return x_paths, y_paths

    def price_european_option(self, T, N, M, K, option_type='call', r=0.0):
        """
        Prices a European option using Monte Carlo simulation.

        Parameters:
        T, N, M: Simulation parameters.
        K (float): Strike price.
        option_type (str): 'call' or 'put'.
        r (float): Risk-free interest rate.

        Returns:
        price (float): The estimated option price.
        """
        x_paths, _ = self.simulate_paths(T, N, M)
        
        # Get the log-price at maturity T
        x_T = x_paths[:, -1]
        
        # Calculate asset price at maturity S_T
        S0 = np.exp(self.x0)
        # Note: The simulation is for the log-price itself (x_t), not the log-return.
        # So S_T = exp(x_T). The paper sets S_0=1 (x_0=0) for convenience. [cite: 562]
        # For a general S0, S_t = S0 * exp(x_t - x0). Let's assume x0 = log(S0)
        S_T = np.exp(x_T)

        # Calculate option payoff
        if option_type == 'call':
            payoff = np.maximum(S_T - K, 0)
        elif option_type == 'put':
            payoff = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # Discount the average payoff to get the price
        price = np.exp(-r * T) * np.mean(payoff)
        return price

if __name__ == '__main__':
    # --- INPUTS ---
    # Model M1 parameters from Table 2 
    # Note: For the scalar case (p=1), parameters are numbers, but we use 1x1 arrays.
    # p = 1
    # Lambda_M1 = np.array([[6.0]])
    # b_M1 = np.array([[1.0]]) # Not specified in table, assumed to be 1 for scalar case.
    # alpha_M1 = 0.0064
    # beta_M1 = np.array([[0.0]])
    # Gamma_M1 = np.array([[3.6334]])

    Lambda_M1 = np.array([[0.45]])
    b_M1 = np.array([[-0.5474]]) # Not specified in table, assumed to be 1 for scalar case.
    alpha_M1 = 0.2704
    beta_M1 = np.array([[2.06]])
    Gamma_M1 = np.array([[0.0062]])
    
    # Initial state (a neutral starting point)
    #S0 = 100
    #S0 = 395.94
    S0 = 420
    x0_M1 = np.log(S0)
    y0_M1 = np.array([[0.0]])
    
    # Simulation parameters
    # T = 1.0  # 1 year maturity
    #T = 0.038 
    T = 0.04
    N = 250  # Daily steps, as used in the paper 
    M = 100000 # Number of paths
    
    # --- IMPLEMENTATION ---
    qhr_model = QHRModel(Lambda_M1, b_M1, alpha_M1, beta_M1, Gamma_M1, x0_M1, y0_M1)

    # 1. Simulate asset price paths
    print("Simulating paths...")
    x_paths_sim, y_paths_sim = qhr_model.simulate_paths(T, N, M)
    S_paths_sim = np.exp(x_paths_sim)
    print("Simulation complete.")

    # 2. Price an at-the-money call option
    K = S0 # At-the-money
    call_price = qhr_model.price_european_option(T, N, M, K=K, option_type='call')
    print(f"Price of a 1-year ATM call option with strike {K:.2f}: {call_price:.4f}")
    
    # # --- OUTPUT ---
    # # Plot a few simulated price paths
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.linspace(0, T, N + 1), S_paths_sim[:50, :].T)
    # plt.title(f'QHR Model: First 50 Simulated Price Paths for S0={S0}')
    # plt.xlabel('Time (Years)')
    # plt.ylabel('Asset Price')
    # plt.grid(True)
    # plt.show()