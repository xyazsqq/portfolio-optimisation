import numpy as np
from scipy.optimize import minimize


# Known example: 3-asset case with hand-derived closed form solution
# Used to verify python solver
Sigma = np.array([[1, 1, 0],
                   [1, 2, 1],
                   [0, 1, 3]])
mu = np.array([1, 2, 3])
Rf = 1

def min_variance_for_return(Sigma, mu, r):
    """
    Find minimum-variance for long-only portfolio weights that achieve
    a target expected return r, given covariance matrix Sigma and expected
    returns mu.
    """
    
    n = len(mu)
    
    def objective(w):
        return w @ Sigma @ w
    
    def constraint_budget(w):
        return sum(w) - 1
    
    def constraint_return(w):
        return w @ mu - r
    
    constraints = [
        {'type': 'eq', 'fun': constraint_budget},
        {'type': 'eq', 'fun': constraint_return}
    ]
      
    bounds = [(0, None) for _ in range(n)]
    x0 = np.ones(n)/n
    
    result = minimize(
        objective, x0, method = 'SLSQP',
        bounds = bounds, constraints = constraints
    )
    
    if not result.success:
        raise RuntimeError(f"Optimisation failed at r = {r}: {result.message}")
    
    return result.x


def sharpe_ratio(w, mu, Sigma, Rf):
    """
    Compute Sharpe ratio of portfolio with weights w,
    given expected returns mu, covariance Sigma, and risk-free rate Rf.
    """
    
    portfolio_return = w @ mu
    variance = w @ Sigma @ w
    std_dev = np.sqrt(variance)
    return (portfolio_return - Rf) / std_dev


def find_max_sharpe(Sigma, mu, Rf, r_min, r_max, steps = 1000):
    """
    Grid-search over target returns in [r_min, r_max] to find the
    long-only portfolio with highest Sharpe ratio.
    This searchs the full feasible region for solutions that closed-form 
    Lagrangian derivation may miss.
    """
    
    best_sharpe = -np.inf
    best_r = None
    best_weights = None
    
    for r in np.linspace(r_min, r_max, steps):
        weights = min_variance_for_return(Sigma, mu, r)
        sr = sharpe_ratio(weights, mu, Sigma, Rf)
        
        if sr > best_sharpe:
            best_sharpe = sr
            best_r = r
            best_weights = weights
        
    return best_r, best_weights, best_sharpe

if __name__ == "__main__":
    print("3-asset weights at r=8/3:", min_variance_for_return(Sigma, mu, 8 / 3))
    print("Sharpe at known point (0, 1/3, 2/3):",
          sharpe_ratio(np.array([0, 1 / 3, 2 / 3]), mu, Sigma, Rf))
    print("Max-Sharpe search (full feasible range):",
          find_max_sharpe(Sigma, mu, Rf, mu.min(), mu.max()))
    
