import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.fft import ifft, fftshift
import matplotlib.pyplot as plt

# ---
# COMPONENT 1: DATA SIMULATION
# ---

def simulate_gvm_samples(n_samples, theta, sigma, nu, dt=1.0):
    """
    Simulates samples from a Gaussian Variance-Mean (GVM) mixture,
    which corresponds to the increments of a Variance-Gamma (VG) process.
    
    This implements the model: X = theta*V + sigma*sqrt(V)*Z
    where V is the Gamma-distributed "market clock" increment[cite: 68].
    
    Args:
        n_samples (int): Number of log-return samples to generate.
        theta (float): The drift parameter.
        sigma (float): The volatility parameter.
        nu (float): The variance rate of the Gamma process.
        dt (float): The time step (e.g., 1/252 for one day).
        
    Returns:
        np.ndarray: An array of simulated log-returns.
    """
    # The paper's parameterization for the Gamma subordinator (market clock)
    # is E[tau_t] = t[cite: 191]. This implies mu=1.
    # For a Gamma(a, scale) dist, mean = a*scale, var = a*scale**2
    # We set mean = dt and var = nu*dt.
    # This gives: scale = var/mean = (nu*dt)/dt = nu
    # And shape 'a' = mean/scale = dt/nu
    shape_a = dt / nu
    scale_s = nu
    
    # 1. Simulate the "market clock" (subordinator) increments V
    # These are Gamma(shape=dt/nu, scale=nu) [cite: 190, 237]
    V = np.random.gamma(shape=shape_a, scale=scale_s, size=n_samples)
    
    # 2. Simulate the standard Normal increments Z
    Z = np.random.normal(size=n_samples)
    
    # 3. Combine them using the GVM mixture formula [cite: 68]
    X_t = theta * V + sigma * np.sqrt(V) * Z
    
    return X_t

# ---
# COMPONENT 2: THE TIME-CHANGE TRANSFORM
# ---

def time_change_transform(log_returns, theta=0.0, n_grid=2**12, tau_max=0.1):
    """
    Implements the paper's "Time-Change Transform" (Definition 6)
    to find the empirical density of the subordinator (market clock).
    
    It maps the density of returns f_X(x) to the density of the 
    subordinator f_tau(tau) via Fourier analysis[cite: 273, 278].
    
    This implementation assumes sigma=1, as per Definition 6[cite: 273].
    
    Args:
        log_returns (np.ndarray): Array of observed log-returns.
        theta (float): The drift parameter, assumed to be known.
        n_grid (int): The number of points for the FFT grid.
        tau_max (float): The maximum value for the 'tau' grid.
        
    Returns:
        tuple: (tau_grid, f_tau_empirical)
               - tau_grid: The grid on which the density is evaluated.
               - f_tau_empirical: The empirical probability density of the subordinator.
    """
    
    # 1. Set up the FFT grids for tau (time) and omega (frequency)
    d_tau = tau_max / n_grid
    tau_grid = np.arange(n_grid) * d_tau
    
    d_omega = 2 * np.pi / (n_grid * d_tau)
    # Create an FFT-ordered frequency grid: [0, ..., max, -max, ..., -min]
    omega_grid = d_omega * np.concatenate((np.arange(0, n_grid/2), 
                                           np.arange(-n_grid/2, 0)))
    
    # 2. Calculate the "clock's" characteristic function (psi_tau)
    # This is the core formula from the paper's transform:
    # psi_tau(omega) = E[exp( (-theta + sqrt(theta^2 + 2i*omega)) * X )] [cite: 274]
    
    # Reshape for numpy broadcasting
    log_returns_col = log_returns.reshape(-1, 1)
    
    # The complex argument for the expectation
    complex_arg = -theta + np.sqrt(theta**2 + 2j * omega_grid)
    
    # Calculate the expectation as the sample mean
    psi_tau_omega = np.mean(np.exp(complex_arg * log_returns_col), axis=0)
    
    # 3. Invert psi_tau to get the density f_tau
    # f_tau(tau) = (1/2pi) * Integral[ psi_tau(omega) * exp(-i*omega*tau) d_omega] [cite: 148]
    # We use the inverse FFT, which calculates:
    # ifft(Y)[k] = (1/N) * Sum[ Y[j] * exp(2*pi*i*j*k / N) ]
    # To match the inversion integral, we use ifft(Y) * N
    
    f_tau_empirical = (ifft(psi_tau_omega) * n_grid).real
    
    # We only return the positive side of the time grid
    return tau_grid, f_tau_empirical

# ---
# COMPONENT 3: PARAMETER FITTING
# ---

def fit_gamma_to_density(tau_grid, f_tau, dt=1.0):
    """
    Fits a Gamma distribution (shape, scale) to the empirical density
    obtained from the time-change transform.
    
    Args:
        tau_grid (np.ndarray): The 'tau' grid.
        f_tau (np.ndarray): The empirical density on that grid.
        dt (float): The time step of the original data.
        
    Returns:
        tuple: (shape_fit, scale_fit, nu_fit)
               - shape_fit (a): The fitted Gamma shape parameter.
               - scale_fit (s): The fitted Gamma scale parameter.
               - nu_fit (float): The 'nu' parameter derived from the fit.
    """
    # Normalize the empirical density to ensure it integrates to 1
    d_tau = tau_grid[1] - tau_grid[0]
    f_tau_normalized = f_tau / (np.sum(f_tau) * d_tau)
    
    # We need to find the Gamma(a, s) parameters that best fit this density.
    # We use optimization to minimize the squared error.
    
    def error_func(params):
        shape_a, scale_s = params
        if shape_a <= 0 or scale_s <= 0:
            return 1e9 # Enforce positive parameters
        
        theoretical_pdf = stats.gamma.pdf(tau_grid, a=shape_a, scale=scale_s)
        return np.sum((f_tau_normalized - theoretical_pdf)**2)
    
    # Use empirical mean/variance for a good initial guess
    # (Filter out negative tau values from the grid if any)
    positive_mask = tau_grid > 0
    tau_pos = tau_grid[positive_mask]
    f_tau_pos = f_tau_normalized[positive_mask]
    
    mean_emp = np.sum(tau_pos * f_tau_pos) * d_tau
    var_emp = np.sum(((tau_pos - mean_emp)**2) * f_tau_pos) * d_tau
    
    # Guesses from method of moments
    scale_guess = var_emp / mean_emp
    shape_guess = mean_emp / scale_guess
    
    # Run the optimization
    result = minimize(error_func, [shape_guess, scale_guess], 
                      method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None)])
    
    shape_fit, scale_fit = result.x
    
    # From our model setup (see simulate_gvm_samples):
    # scale = nu  and  shape = dt / nu
    # We can get nu from both parameters as a consistency check
    nu_fit_from_scale = scale_fit
    nu_fit_from_shape = dt / shape_fit
    
    print(f"[Fit Check] nu from scale: {nu_fit_from_scale:.5f}, nu from shape: {nu_fit_from_shape:.5f}")
    
    return shape_fit, scale_fit, nu_fit_from_scale

# ---
# COMPONENT 4: OPTION PRICING
# ---

def price_vg_call_mc(S0, K, T, r, q, sigma, nu, theta, n_sims=100000):
    """
    Prices a European Call Option using the Variance-Gamma model
    via Monte Carlo simulation.
    
    This directly simulates the risk-neutral S_T.
    
    Args:
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        sigma (float): VG volatility parameter.
        nu (float): VG variance rate parameter.
        theta (float): VG drift parameter.
        n_sims (int): Number of Monte Carlo paths.
        
    Returns:
        float: The estimated price of the call option.
    """
    
    # 1. Risk-Neutrality: We need E[S_T] = S0 * e^((r-q)*T).
    # The S_T is S0 * exp((r-q)T + X_T + omega*T)
    # where X_T is the "pure" VG process and omega is the convexity correction.
    # omega = (1/nu) * log(1 - theta*nu - 0.5*sigma^2*nu)
    # This ensures E[exp(X_T + omega*T)] = 1.
    omega = (1/nu) * np.log(1 - theta*nu - 0.5 * sigma**2 * nu)
    
    # 2. Simulate the "market clock" at maturity T
    # tau_T ~ Gamma(shape=T/nu, scale=nu)
    shape_a = T / nu
    scale_s = nu
    tau_T = np.random.gamma(shape=shape_a, scale=scale_s, size=n_sims)
    
    # 3. Simulate the Brownian motion component
    Z = np.random.normal(size=n_sims)
    
    # 4. Build the terminal log-price component X_T 
    X_T = theta * tau_T + sigma * np.sqrt(tau_T) * Z
    
    # 5. Calculate the terminal stock price S_T
    S_T = S0 * np.exp((r - q + omega) * T + X_T)
    
    # 6. Calculate payoffs and discount to find the price
    payoffs = np.maximum(S_T - K, 0)
    price = np.mean(payoffs) * np.exp(-r * T)
    
    return price

# ---
# MAIN EXAMPLE USE CASE
# ---

if __name__ == "__main__":
    
    print("--- 1. SETUP ---")
    # Define "ground truth" parameters for our simulation
    # We set theta=0 and sigma=1, as assumed by the paper's
    # empirical transform [cite: 273, 377]
    TRUE_THETA = 0.0
    TRUE_SIGMA = 1.0 
    TRUE_NU = 0.1
    
    # Data properties
    N_SAMPLES = 50000     # Number of 'days' to simulate
    DT = 1.0 / 252.0      # Time step (1 trading day)
    
    print(f"Ground Truth Parameters: theta={TRUE_THETA}, sigma={TRUE_SIGMA}, nu={TRUE_NU}")

    # ---
    print("\n--- 2. DATA SIMULATION ---")
    print(f"Simulating {N_SAMPLES} log-returns from the VG process...")
    log_returns = simulate_gvm_samples(N_SAMPLES, TRUE_THETA, TRUE_SIGMA, TRUE_NU, dt=DT)
    print(f"Simulation complete. Mean log-return: {np.mean(log_returns):.6f}")

    # ---
# --- 3. APPLYING TIME-CHANGE TRANSFORM ---
    print("\n--- 3. APPLYING TIME-CHANGE TRANSFORM ---")
    print("Applying the paper's transform to find the hidden 'market clock' density...")
    
    # FIX: Normalize the log-returns to have std_dev = 1
    # This matches the transform's assumption (sigma=1) and fixes numerical instability
    log_returns_normalized = log_returns / np.std(log_returns)

    # We need a 'tau_max' for the grid. The mean should be dt (0.004)
    # and variance nu*dt (0.0004). A max of 0.02 should be sufficient.
    tau_grid, f_tau_empirical = time_change_transform(log_returns_normalized,  # <-- Use the normalized data
                                                      theta=TRUE_THETA, 
                                                      n_grid=2**12, 
                                                      tau_max=0.02)
    print("Transform complete.")

    # ---
    print("\n--- 4. FITTING EMPIRICAL DENSITY ---")
    print("Fitting a theoretical Gamma distribution to the empirical density...")
    shape_fit, scale_fit, nu_fit = fit_gamma_to_density(tau_grid, f_tau_empirical, dt=DT)
    
    print("\n--- FIT RESULTS ---")
    print(f"  Ground Truth nu: {TRUE_NU:.5f}")
    print(f"  FITTED nu:       {nu_fit:.5f}")
    
    # ---
    print("\n--- 5. PLOTTING RESULTS ---")
    print("Plotting empirical density vs. fitted theoretical density...")
    
    # Normalize empirical density for plotting
    d_tau = tau_grid[1] - tau_grid[0]
    f_tau_norm = f_tau_empirical / (np.sum(f_tau_empirical) * d_tau)
    
    # Get theoretical PDF from fitted parameters
    theoretical_pdf = stats.gamma.pdf(tau_grid, a=shape_fit, scale=scale_fit)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_grid, f_tau_norm, label='Empirical Density (from Transform)', lw=2)
    plt.plot(tau_grid, theoretical_pdf, label=f'Fitted Gamma (nu={nu_fit:.4f})', 
             linestyle='--', color='red', lw=2)
    plt.title("Paper's Time-Change Transform: Empirical vs. Fitted Density")
    plt.xlabel("Subordinator Value (tau)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 0.015) # Zoom in on the main part of the density
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    # ---
    print("\n--- 6. OPTION PRICING EXAMPLE ---")
    print("Using our FITTED 'nu' parameter to price an option...")
    
    # Option parameters
    S0 = 259.58 # Initial stock price
    K = 255 # Strike price
    T = 1/365  # 6 months is 0.5 years
    r = 0.03565 # Risk-free rate
    q = 0.004 # Continuous dividend yield
    
    # Use the parameters we found/assumed
    # (sigma=1 and theta=0 are assumed by the transform method)
    SIGMA_PARAM = 1.0 
    THETA_PARAM = 0.0
    NU_PARAM = nu_fit # The parameter we discovered!
    
    price_fitted = price_vg_call_mc(S0, K, T, r, q, 
                                    SIGMA_PARAM, NU_PARAM, THETA_PARAM)
    
    print("\n--- PRICING RESULTS ---")
    print(f"  Option Price (using FITTED nu={NU_PARAM:.5f}):   ${price_fitted:.4f}")
    
    # For comparison, let's price with the "ground truth" parameters
    price_true = price_vg_call_mc(S0, K, T, r, q, 
                                  TRUE_SIGMA, TRUE_NU, TRUE_THETA)
    print(f"  Option Price (using TRUE nu={TRUE_NU:.5f}):       ${price_true:.4f}")