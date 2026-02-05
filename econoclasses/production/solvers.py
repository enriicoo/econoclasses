"""
Solvers for cost minimization and profit maximization.

Mirrors preferences/solvers.py structure.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class CostSolution:
    """Result of cost minimization."""
    K: float
    L: float
    cost: float
    output: float
    is_interior: bool
    method: str
    
    def as_tuple(self) -> Tuple[float, float]:
        return (self.K, self.L)


@dataclass
class ProfitSolution:
    """Result of profit maximization."""
    K: float
    L: float
    Q: float
    revenue: float
    cost: float
    profit: float
    method: str


# =============================================================================
# COST MINIMIZATION SOLVERS
# =============================================================================

def cost_min_cobb_douglas(alpha: float, A: float, w: float, r: float, Q_target: float) -> CostSolution:
    """
    Cost minimization for Cobb-Douglas: Q = A K^α L^(1-α)
    
    Solution: K/L = [α/(1-α)] × (w/r)
    """
    beta = 1 - alpha
    
    # From FOC: K/L = (α/β)(w/r)
    # From constraint: A K^α L^β = Q
    
    # Solve: L = [Q/A]^(1/(α+β)) × [α/β × w/r]^(-α/(α+β))
    # For CRS (α+β=1): L = (Q/A) × [(1-α)/α × r/w]^α
    
    ratio = (alpha / beta) * (w / r)
    L_star = (Q_target / A) * (ratio ** (-alpha))
    K_star = ratio * L_star
    
    cost = r * K_star + w * L_star
    
    return CostSolution(K_star, L_star, cost, Q_target, is_interior=True, method='analytical')


def cost_min_cobb_douglas_general(alpha: float, beta: float, A: float, 
                                   w: float, r: float, Q_target: float) -> CostSolution:
    """
    Cost minimization for general Cobb-Douglas: Q = A K^α L^β
    """
    # K/L = (α/β)(w/r)
    ratio = (alpha / beta) * (w / r)
    
    # From constraint: A K^α L^β = Q
    # K = ratio × L, so: A (ratio×L)^α L^β = Q
    # L^(α+β) = Q / (A × ratio^α)
    
    L_star = (Q_target / (A * (ratio ** alpha))) ** (1 / (alpha + beta))
    K_star = ratio * L_star
    cost = r * K_star + w * L_star
    
    return CostSolution(K_star, L_star, cost, Q_target, is_interior=True, method='analytical')


def cost_min_ces(alpha: float, rho: float, A: float,
                 w: float, r: float, Q_target: float) -> CostSolution:
    """
    Cost minimization for CES: Q = A [αK^ρ + (1-α)L^ρ]^(1/ρ)
    """
    # Handle special cases
    if abs(rho) < 1e-10:
        return cost_min_cobb_douglas(alpha, A, w, r, Q_target)
    
    if rho <= -50:
        return cost_min_leontief(alpha, 1 - alpha, A, w, r, Q_target)
    
    sigma = 1 / (1 - rho)
    
    # FOC: K/L = [α/(1-α)]^σ × (w/r)^σ
    ratio = ((alpha / (1 - alpha)) ** sigma) * ((w / r) ** sigma)
    
    # From constraint, solve for L
    # Numerical approach for general case
    def objective(L):
        K = ratio * L
        Q = A * (alpha * K**rho + (1 - alpha) * L**rho) ** (1/rho)
        return (Q - Q_target) ** 2
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(0.001, 1000), method='bounded')
    
    L_star = result.x
    K_star = ratio * L_star
    cost = r * K_star + w * L_star
    
    return CostSolution(K_star, L_star, cost, Q_target, is_interior=True, method='analytical')


def cost_min_leontief(alpha: float, beta: float, A: float,
                      w: float, r: float, Q_target: float) -> CostSolution:
    """
    Cost minimization for Leontief: Q = A × min(αK, βL)
    
    At optimum: αK = βL = Q/A
    """
    K_star = Q_target / (A * alpha)
    L_star = Q_target / (A * beta)
    cost = r * K_star + w * L_star
    
    return CostSolution(K_star, L_star, cost, Q_target, is_interior=True, method='analytical')


def cost_min_linear(alpha: float, beta: float, A: float,
                    w: float, r: float, Q_target: float) -> CostSolution:
    """
    Cost minimization for linear: Q = A(αK + βL)
    
    Corner solution: use only the cheaper input per unit output.
    """
    cost_per_Q_via_K = r / (A * alpha)
    cost_per_Q_via_L = w / (A * beta)
    
    if cost_per_Q_via_K < cost_per_Q_via_L:
        K_star = Q_target / (A * alpha)
        L_star = 0.0
    elif cost_per_Q_via_K > cost_per_Q_via_L:
        K_star = 0.0
        L_star = Q_target / (A * beta)
    else:
        # Indifferent - split evenly
        K_star = Q_target / (2 * A * alpha)
        L_star = Q_target / (2 * A * beta)
    
    cost = r * K_star + w * L_star
    return CostSolution(K_star, L_star, cost, Q_target, is_interior=False, method='analytical')


def cost_min_numerical(production_func: Callable[[float, float], float],
                       w: float, r: float, Q_target: float) -> CostSolution:
    """
    Numerical cost minimization for any production function.
    """
    def cost(inputs):
        K, L = inputs
        return r * K + w * L
    
    def output_constraint(inputs):
        K, L = inputs
        try:
            return production_func(K, L) - Q_target
        except:
            return -1e10
    
    x0 = (Q_target / 2, Q_target / 2)
    result = minimize(cost, x0, method='SLSQP',
                      constraints={'type': 'eq', 'fun': output_constraint},
                      bounds=[(0.001, None), (0.001, None)])
    
    K_star, L_star = result.x
    total_cost = cost(result.x)
    
    return CostSolution(K_star, L_star, total_cost, Q_target,
                        is_interior=(K_star > 1e-6 and L_star > 1e-6),
                        method='numerical')


# =============================================================================
# PROFIT MAXIMIZATION SOLVERS
# =============================================================================

def profit_max_cobb_douglas(alpha: float, A: float, 
                            w: float, r: float, P: float) -> ProfitSolution:
    """
    Profit maximization for CRS Cobb-Douglas.
    
    Note: With CRS, if P > AC then profit → ∞, if P < AC then Q* = 0.
    Returns the marginal cost condition solution.
    """
    beta = 1 - alpha
    
    # For interior solution, use the cost function approach
    # MC = w^β r^α / (A α^α β^β)
    MC = (w ** beta) * (r ** alpha) / (A * (alpha ** alpha) * (beta ** beta))
    
    if P < MC * 0.999:
        # Price below MC, produce zero
        return ProfitSolution(0, 0, 0, 0, 0, 0, method='analytical')
    
    # With CRS, at P = MC any output is "optimal" - return a reasonable scale
    # Use P = MC to find output level where firm breaks even
    # This is a simplification - CRS implies zero profit at equilibrium
    
    # Assume a target output for demonstration
    Q_star = 10  # Scale parameter
    
    cost_sol = cost_min_cobb_douglas(alpha, A, w, r, Q_star)
    K_star, L_star = cost_sol.K, cost_sol.L
    
    revenue = P * Q_star
    cost = cost_sol.cost
    profit = revenue - cost
    
    return ProfitSolution(K_star, L_star, Q_star, revenue, cost, profit, method='analytical')


def profit_max_cobb_douglas_general(alpha: float, beta: float, A: float,
                                     w: float, r: float, P: float) -> ProfitSolution:
    """
    Profit maximization for general Cobb-Douglas: Q = A K^α L^β
    
    With DRS (α+β < 1), there's a finite optimal output.
    """
    gamma = alpha + beta
    
    if gamma >= 1:
        # CRS or IRS: use simplified approach
        return profit_max_cobb_douglas(alpha / gamma, A, w, r, P)
    
    # DRS: Unique interior solution
    # From FOCs: P × MP_K = r, P × MP_L = w
    # MP_K = α A K^(α-1) L^β = α Q / K
    # MP_L = β A K^α L^(β-1) = β Q / L
    
    # K = α P Q / r
    # L = β P Q / w
    # Substituting into production function:
    # Q = A (αPQ/r)^α (βPQ/w)^β
    # Q^(1-γ) = A (αP/r)^α (βP/w)^β
    
    coef = A * ((alpha * P / r) ** alpha) * ((beta * P / w) ** beta)
    Q_star = coef ** (1 / (1 - gamma))
    
    K_star = alpha * P * Q_star / r
    L_star = beta * P * Q_star / w
    
    revenue = P * Q_star
    cost = r * K_star + w * L_star
    profit = revenue - cost
    
    return ProfitSolution(K_star, L_star, Q_star, revenue, cost, profit, method='analytical')


def profit_max_numerical(production_func: Callable[[float, float], float],
                         w: float, r: float, P: float,
                         K_max: float = 100, L_max: float = 100) -> ProfitSolution:
    """
    Numerical profit maximization.
    """
    def neg_profit(inputs):
        K, L = inputs
        try:
            Q = production_func(K, L)
            return -(P * Q - r * K - w * L)
        except:
            return 1e10
    
    x0 = (K_max / 4, L_max / 4)
    result = minimize(neg_profit, x0, method='L-BFGS-B',
                      bounds=[(0.001, K_max), (0.001, L_max)])
    
    K_star, L_star = result.x
    Q_star = production_func(K_star, L_star)
    revenue = P * Q_star
    cost = r * K_star + w * L_star
    profit = revenue - cost
    
    return ProfitSolution(K_star, L_star, Q_star, revenue, cost, profit, method='numerical')


# =============================================================================
# REGISTRIES
# =============================================================================

COST_MIN_SOLVERS = {
    'cobb-douglas': cost_min_cobb_douglas,
    'cobb-douglas-general': cost_min_cobb_douglas_general,
    'ces': cost_min_ces,
    'leontief': cost_min_leontief,
    'linear': cost_min_linear,
}

PROFIT_MAX_SOLVERS = {
    'cobb-douglas': profit_max_cobb_douglas,
    'cobb-douglas-general': profit_max_cobb_douglas_general,
}
