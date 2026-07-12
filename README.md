# Don't Put All Your Eggs In One Basket!
-> ***Portfolio Optimisation via Lagrange Multipliers***

Finding the optimal long-only portfolio for a given return target. Derived by hand, then solved and verified numerically in Python.


## Problem Setup

Given a covariance matrix $\Sigma$ and a vector of expected returns $\mu$ across $n$ assets, we want to find portfolio weights $\lambda$ that minimise portfolio variance:

$$\large \sigma^2 = \lambda^T \Sigma \lambda$$

Here, variance is used as the measure of portfolio risk, consistent with mean-variance portfolio theory (Markowitz). 

*Constraints:*

$$\large \sum_{i=1}^n \lambda_i = 1 \quad \text{(weights sum to 1)}$$

$$\large \lambda^T \mu = r \quad \text{(hits target expected return } r\text{)}$$

$$\large \lambda_i \geq 0 \quad \text{for all } i \quad \text{(no shorting)}$$

The Sharpe ratio of a resulting portfolio, relative to a risk-free rate $R_f$ is:

$$\large Sharpe = \frac{r - R_f}{\sigma}$$

$R_f$ represents the return available from a risk-free alternative. What you'd earn without taking on any investment risk at all, such as cash in a savings account.

## Method

**Closed-form derivation.** Ignoring the non-negativity constraint, this is solvable via Lagrange multipliers. Defining $A = \mathbf{1}^T \Sigma^{-1} \mathbf{1}$, $B = \mathbf{1}^T \Sigma^{-1} \mu$, $C = \mu^T \Sigma^{-1} \mu$, the optimal weights at a given $r$ are:

$$\large \lambda = \gamma \, \Sigma^{-1}\mathbf{1} + \delta \, \Sigma^{-1}\mu$$

where $\gamma, \delta$ solve a 2×2 linear system in $A, B, C, r$. This gives an *interior* solution — valid only where every $\lambda_i$ it produces happens to stay non-negative.

**Numerical solution.** `min_variance_for_return()` solves the full constrained problem (including $\lambda_i \geq 0$) directly using `scipy.optimize.minimize` (SLSQP). Since $\Sigma$ is positive semi-definite, the problem is convex, so the solver is guaranteed to find the global optimum regardless of starting point.

`find_max_sharpe()` searches across a range of target returns to find the portfolio with the best risk-adjusted return, rather than assuming the maximum lies at the edge of the closed-form solution's valid range.

## Worked Example

Three-asset case:

$$\large \Sigma = \begin{bmatrix} 1 & 1 & 0 \\\\ 1 & 2 & 1 \\\\ 0 & 1 & 3 \end{bmatrix}, \quad \mu = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}^T, \quad R_f = 1$$

The closed-form derivation is valid for $r \in [2, 8/3]$ (the range over which all three weights stay non-negative) giving an interior optimum at:  
 -> $r = 8/3$  
 -> $\lambda (weights) = [0, \frac{1}{3}, \frac{2}{3}]$  
 -> Sharpe ratio $\approx 1.178$.

## Evaluation and Limitations

Searching the full feasible range using Python produced a portfolio with a higher Sharpe ratio than the hand-derived closed-form solution. Why?  

The hand derivation solves only the equality constraints (budget and target return), ignoring $\lambda_i \geq 0$ while solving. 
It only checks afterwards whether the answer happens to satisfy that condition. This closed-form result was plotted in Desmos to visualise the 
efficient frontier and identify the maximum-Sharpe point graphically, 
which is where $r = 8/3$ came from originally.

This is why it's only valid on $[2, 8/3]$: the range where the unconstrained answer happens to land on non-negative weights by chance, not by design.

Past $r = 8/3$, the unconstrained maths would force $\lambda_1$ to go negative (shorting) which isn't incorporated in this model, so it isn't a valid answer for this scenario.

When an asset can't go negative, the constraint $\lambda_1 \geq 0$ becomes binding: $\lambda_1$ is pinned at exactly 0. Once $\lambda_1 = 0$ is fixed
the problem effectively becomes a new 2-asset problem, reallocating only between assets 2 and 3. A new closed-form derivation 
restricted to just these two assets reproduces the solver's exact answer.

***Conclusion:*** A closed-form Lagrangian only solves one segment of the true, piecewise long-only efficient frontier. 
Getting the full picture requires either deriving a separate closed-form solution for every possible combination of active/inactive constraints (tedious and error-prone as $n$ grows), 
or using a numerical solver that handles inequality constraints directly.



## Generalisation to n Assets

`min_variance_for_return()` and `find_max_sharpe()` place no assumptions on $n$ anywhere in their implementation — bounds, starting weights, and constraints are all built from `len(mu)`.

## Setup
```bash
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash); use venv\Scripts\activate for PowerShell
pip install -r requirements.txt
python main.py
```

## Files

- `main.py` — core optimisation functions and worked example
- `requirements.txt` — pinned dependencies (numpy, scipy)
