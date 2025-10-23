import numpy as np
import sys

def price_american_option(
    S0: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    q: float, 
    option_type: str, 
    num_steps: int,
    cash_dividends: list[tuple[float, float]]
) -> tuple[float, float]:
    """
    Prices an American option with discrete cash dividends and a continuous proportional yield.
    This uses the "Shift" (or "Modified") Binomial Tree method.

    Args:
        S0: Initial stock price.
        K: Strike price.
        T: Time to maturity (in years).
        r: Risk-free interest rate.
        sigma: Volatility.
        q: Continuous dividend yield (represents proportional dividends).
        option_type: 'put' or 'call'.
        num_steps: Number of steps in the binomial tree.
        cash_dividends: A list of (amount, ex_dividend_time) tuples.
                        e.g., [(1.0, 0.25), (1.0, 0.75)] for two $1 dividends.
                        
    Returns:
        A tuple of (option_price, immediate_exercise_value)
    """
    
    # --- 1. Calculate PV of all future cash dividends ---
    # This is subtracted from S0 to get the "stochastic" part
    try:
        pv_cash_dividends = sum(
            [amount * np.exp(-r * time) for amount, time in cash_dividends if time > 0 and time <= T]
        )
    except Exception as e:
        print(f"Error calculating dividend PV: {e}", file=sys.stderr)
        return 0, 0
        
    S_star = S0 - pv_cash_dividends
    if S_star <= 0:
        print(f"Warning: S0 ({S0}) is less than PV of dividends ({pv_cash_dividends}). Model may be inaccurate.", file=sys.stderr)
        # Price is likely just intrinsic value if S_star is non-positive
        S_star = 0.01 

    # --- 2. Set up Binomial Tree parameters (Cox-Ross-Rubinstein) ---
    dt = T / num_steps
    u = np.exp(sigma * np.sqrt(dt))  # Up-factor
    d = 1.0 / u                      # Down-factor
    
    # Risk-neutral probability
    p_u = (np.exp((r - q) * dt) - d) / (u - d)
    p_d = 1.0 - p_u
    
    if p_u < 0 or p_u > 1:
        print(f"Warning: Arbitrage possibility detected. p_u = {p_u:.4f}. Check inputs.", file=sys.stderr)
        print("Try increasing num_steps or adjusting r, q, sigma.", file=sys.stderr)
        if p_u < 0: p_u = 0.0001
        if p_u > 1: p_u = 0.9999
        p_d = 1.0 - p_u

    # --- 3. Initialize Trees ---
    # We need to store the "stochastic" stock price S* and the option value
    # V_tree[i, j] is the value at time-step i, in state j
    # (j=0 is all "up" moves, j=i is all "down" moves)
    S_star_tree = np.zeros((num_steps + 1, num_steps + 1))
    V_tree = np.zeros((num_steps + 1, num_steps + 1))
    
    # --- 4. Build the S* (Stochastic Price) Tree (Forward Pass) ---
    for i in range(num_steps + 1):
        for j in range(i + 1):
            S_star_tree[i, j] = S_star * (u**(i - j)) * (d**j)

    # --- 5. Price the Option (Backward Pass) ---
    
    # Payoff at maturity (T)
    for j in range(num_steps + 1):
        t = T # We are at maturity
        
        # At maturity, the PV of remaining dividends is 0
        S_actual = S_star_tree[num_steps, j] 
        
        if option_type == 'put':
            V_tree[num_steps, j] = max(K - S_actual, 0.0)
        elif option_type == 'call':
            V_tree[num_steps, j] = max(S_actual - K, 0.0)

    # Iterate backwards from (T - dt) to 0
    for i in range(num_steps - 1, -1, -1):
        t = i * dt # Current time at this step
        
        # Find the present value of cash dividends remaining *after* time t
        pv_remaining_divs = sum(
            [amount * np.exp(-r * (time - t)) for amount, time in cash_dividends if time > t and time <= T]
        )
        
        for j in range(i + 1):
            # A. Calculate the Continuation Value (Hold)
            # This is the discounted expected value of the *next* step
            continuation_value = np.exp(-r * dt) * (
                p_u * V_tree[i + 1, j] + p_d * V_tree[i + 1, j + 1]
            )
            
            # B. Calculate the Exercise Value (Exercise)
            # This is based on the *actual* stock price, not S*
            S_actual = S_star_tree[i, j] + pv_remaining_divs
            
            exercise_value = 0.0
            if option_type == 'put':
                exercise_value = max(K - S_actual, 0.0)
            elif option_type == 'call':
                exercise_value = max(S_actual - K, 0.0)
                
            # C. American Option Value is the max of Hold vs. Exercise
            V_tree[i, j] = max(continuation_value, exercise_value)

    # --- 6. Return the price at t=0 and the immediate exercise value ---
    option_price_t0 = V_tree[0, 0]
    
    # Calculate immediate exercise value at t=0
    S_actual_t0 = S0 # By definition
    immediate_exercise_value = 0.0
    if option_type == 'put':
        immediate_exercise_value = max(K - S_actual_t0, 0.0)
    elif option_type == 'call':
        immediate_exercise_value = max(S_actual_t0 - K, 0.0)

    return option_price_t0, immediate_exercise_value

# --- Main execution block to provide an actionable output ---
if __name__ == "__main__":
    
    # --- 1. Define Model Parameters (YOU CAN CHANGE THESE) ---
    S0_input = 259.83      # Current Stock Price
    K_input = 290    # Strike Price
    T_input = 85/365         # Maturity (in years)
    r_input = 0.03565        # Risk-free rate (e.g., 5%)
    sigma_input = 0.25    # Volatility (e.g., 20%)
    
    # Represents proportional dividends (e.g., 1.5% per year)
    q_input = 0.01     

    cash_div_input = [ (1.03, 0.25),(1.03, 0.50), (1.03, 0.75), (1.03, 1.00) ]
    
    # Represents discrete cash dividends
    # Format: [(amount, time_in_years_to_ex_div_date), ...]
    # cash_div_input = [
    #     (1.0, 0.25),  # $1.00 dividend in 3 months (0.25 years)
    #     (1.0, 0.75)   # $1.00 dividend in 9 months (0.75 years)
    # ]
    
    option_type_input = 'put' # 'put' or 'call'
    steps_input = 2000         # Number of tree steps (increase for accuracy)
    
    # --- 2. Run the Pricer ---
    print("--- American Option Pricer ---")
    print(f"Running model for a {T_input}-year American {option_type_input} option.")
    print(f"Stock: ${S0_input:.2f}, Strike: ${K_input:.2f}, Vol: {sigma_input*100:.0f}%, r: {r_input*100:.1f}%, q: {q_input*100:.1f}%")
    print(f"Cash Dividends: {cash_div_input}")
    print(f"Tree Steps: {steps_input}\n")

    try:
        price, immediate_exercise = price_american_option(
            S0_input,
            K_input,
            T_input,
            r_input,
            sigma_input,
            q_input,
            option_type_input,
            steps_input,
            cash_div_input
        )
        
        # --- 3. Provide Actionable Output ---
        print("--- Result ---")
        print(f"Calculated Option Price: ${price:.4f}")
        print(f"Immediate Exercise Value: ${immediate_exercise:.4f}")
        
        # print("\n--- Actionable Recommendation (at t=0) ---")
        # if price > immediate_exercise:
        #     print(f"HOLD: The option's time value (${price - immediate_exercise:.4f}) is positive.")
        #     print("The model suggests it is not optimal to exercise immediately.")
        # else:
        #     # This can happen (e.g., deep in-the-money put, or a call right before a large dividend)
        #     print(f"EXERCISE IMMEDIATELY: The option price (${price:.4f}) is equal to its intrinsic value.")
        #     print("The model suggests exercising now is optimal.")
            
    except Exception as e:
        print(f"\n--- An error occurred ---", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        print("Please check your input parameters. Arbitrage may be present.", file=sys.stderr)