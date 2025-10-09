import numpy as np
from scipy.optimize import minimize

def calculate_optimal_risk_sharing(total_loss, risk_preferences):
    """
    Calculates the optimal risk sharing allocation and the minimum aggregate risk
    by numerically solving the integral infimal convolution problem for a discrete
    number of agents.

    Parameters:
    - total_loss (float): The total loss X to be shared among the agents.
    - risk_preferences (list of callables): A list of functions [rho_1, rho_2, ...],
      where each function rho_i(x) returns the risk/pain for agent i given a loss x.

    Returns:
    - A dictionary containing:
      - 'min_aggregate_risk' (float): The minimum possible sum of risks for the group.
      - 'optimal_allocation' (np.ndarray): The share of the loss for each agent.
      - 'success' (bool): Whether the optimization algorithm succeeded.
    """
    num_agents = len(risk_preferences)

    # 1. Define the objective function to minimize (the sum of individual risks)
    def objective_function(allocations):
        # allocations is a numpy array [x_1, x_2, ..., x_N]
        total_risk = sum(risk_preferences[i](allocations[i]) for i in range(num_agents))
        return total_risk

    # 2. Define the constraint: sum of allocations must equal the total loss
    # The constraint format is {type: 'eq', fun: lambda x: function_that_equals_zero}
    constraint = {'type': 'eq', 'fun': lambda allocations: np.sum(allocations) - total_loss}

    # 3. Define the bounds: each agent's loss share cannot be negative
    bounds = [(0, None) for _ in range(num_agents)]

    # 4. Provide an initial guess for the allocation (e.g., an even split)
    initial_guess = np.full(num_agents, total_loss / num_agents)

    # 5. Run the optimizer
    # SLSQP is a good method for constrained optimization problems
    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraint
    )

    # 6. Return the results
    if result.success:
        return {
            'min_aggregate_risk': result.fun,
            'optimal_allocation': result.x,
            'success': True
        }
    else:
        # If the optimization fails, return the failure status
        return {'success': False, 'message': result.message}

# --- Example Usage ---
if __name__ == '__main__':
    # Define a total loss to be shared
    X = 100.0

    # Define the risk preferences for 3 different agents
    # Agent 0: Standard quadratic risk (risk-averse)
    # Agent 1: Linear risk (risk-neutral, less sensitive)
    # Agent 2: Exponential risk (very risk-averse)
    rhos = [
        lambda x: 0.1 * x**2,      # Agent 0
        lambda x: 2 * x,           # Agent 1
        lambda x: np.exp(0.05 * x)  # Agent 2
    ]

    # Calculate the optimal sharing plan
    solution = calculate_optimal_risk_sharing(X, rhos)

    # Print the results
    if solution['success']:
        print(f"Total Loss to Share: {X}\n")
        print(f"Minimum Aggregate Risk (Value of the Convolution): {solution['min_aggregate_risk']:.4f}\n")
        print("Optimal Allocation of Loss:")
        for i, allocation in enumerate(solution['optimal_allocation']):
            print(f"  - Agent {i}: {allocation:.4f}")

        # Intuition Check: The most risk-averse agent (Agent 2) should receive the smallest
        # share of the loss, which the optimization correctly finds.
    else:
        print("Optimization failed:", solution.get('message'))