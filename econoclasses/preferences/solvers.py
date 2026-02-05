"""
Demand solvers for utility maximization.

Analytical solutions for standard forms + numerical fallback.
"""

import sympy as sp
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class DemandSolution:
    """Result of a demand optimization."""
    x: float
    y: float
    utility: float
    is_interior: bool
    method: str
    
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def as_dict(self) -> Dict[str, float]:
        return {'X': self.x, 'Y': self.y}


# =============================================================================
# MARSHALLIAN DEMAND SOLVERS
# =============================================================================

def marshallian_cobb_douglas(alpha: float, beta: float,
                              px: float, py: float, income: float) -> DemandSolution:
    share_x = alpha / (alpha + beta)
    share_y = beta / (alpha + beta)
    x_star = share_x * income / px
    y_star = share_y * income / py
    utility = (x_star ** alpha) * (y_star ** beta)
    return DemandSolution(x_star, y_star, utility, is_interior=True, method='analytical')


def marshallian_perfect_substitutes(alpha: float, beta: float,
                                     px: float, py: float, income: float) -> DemandSolution:
    bang_x = alpha / px
    bang_y = beta / py
    
    if bang_x > bang_y:
        x_star, y_star = income / px, 0.0
    elif bang_x < bang_y:
        x_star, y_star = 0.0, income / py
    else:
        x_star = income / (2 * px)
        y_star = income / (2 * py)
    
    utility = alpha * x_star + beta * y_star
    return DemandSolution(x_star, y_star, utility, is_interior=(bang_x == bang_y), method='analytical')


def marshallian_perfect_complements(alpha: float, beta: float,
                                     px: float, py: float, income: float) -> DemandSolution:
    x_star = beta * income / (beta * px + alpha * py)
    y_star = alpha * income / (beta * px + alpha * py)
    utility = min(alpha * x_star, beta * y_star)
    return DemandSolution(x_star, y_star, utility, is_interior=True, method='analytical')


def marshallian_ces(alpha: float, rho: float,
                    px: float, py: float, income: float) -> DemandSolution:
    if abs(rho) < 1e-10:
        return marshallian_cobb_douglas(alpha, 1 - alpha, px, py, income)
    if rho >= 0.999:
        return marshallian_perfect_substitutes(alpha, 1 - alpha, px, py, income)
    if rho <= -50:
        return marshallian_perfect_complements(alpha, 1 - alpha, px, py, income)
    
    sigma = 1 / (1 - rho)
    term_x = (alpha ** sigma) * (px ** (1 - sigma))
    term_y = ((1 - alpha) ** sigma) * (py ** (1 - sigma))
    
    x_star = (alpha ** sigma) * (px ** (-sigma)) / (term_x + term_y) * income
    y_star = ((1 - alpha) ** sigma) * (py ** (-sigma)) / (term_x + term_y) * income
    utility = (alpha * (x_star ** rho) + (1 - alpha) * (y_star ** rho)) ** (1 / rho)
    
    return DemandSolution(x_star, y_star, utility, is_interior=True, method='analytical')


def marshallian_quasilinear(alpha: float, numeraire: str,
                            px: float, py: float, income: float) -> DemandSolution:
    if numeraire == 'X':
        y_interior = alpha * px / py
        cost_y = py * y_interior
        if cost_y <= income:
            y_star = y_interior
            x_star = (income - cost_y) / px
            is_interior = True
        else:
            y_star = income / py
            x_star = 0.0
            is_interior = False
        utility = x_star + alpha * np.log(max(y_star, 1e-10))
    else:
        x_interior = alpha * py / px
        cost_x = px * x_interior
        if cost_x <= income:
            x_star = x_interior
            y_star = (income - cost_x) / py
            is_interior = True
        else:
            x_star = income / px
            y_star = 0.0
            is_interior = False
        utility = alpha * np.log(max(x_star, 1e-10)) + y_star
    
    return DemandSolution(x_star, y_star, utility, is_interior, method='analytical')


def marshallian_stone_geary(alpha: float, beta: float, gamma_x: float, gamma_y: float,
                            px: float, py: float, income: float) -> DemandSolution:
    subsistence_cost = px * gamma_x + py * gamma_y
    if income <= subsistence_cost:
        return DemandSolution(gamma_x, gamma_y, 0.0, is_interior=False, method='analytical')
    
    supernumerary = income - subsistence_cost
    share_x = alpha / (alpha + beta)
    share_y = beta / (alpha + beta)
    
    x_star = gamma_x + share_x * supernumerary / px
    y_star = gamma_y + share_y * supernumerary / py
    utility = ((x_star - gamma_x) ** alpha) * ((y_star - gamma_y) ** beta)
    
    return DemandSolution(x_star, y_star, utility, is_interior=True, method='analytical')


# =============================================================================
# HICKSIAN DEMAND SOLVERS
# =============================================================================

def hicksian_cobb_douglas(alpha: float, beta: float,
                          px: float, py: float, u_target: float) -> DemandSolution:
    ratio = (beta / alpha) * (px / py)
    exponent = 1 / (alpha + beta)
    x_star = (u_target ** exponent) * (ratio ** (-beta * exponent))
    y_star = ratio * x_star
    return DemandSolution(x_star, y_star, u_target, is_interior=True, method='analytical')


def hicksian_perfect_substitutes(alpha: float, beta: float,
                                  px: float, py: float, u_target: float) -> DemandSolution:
    cost_x = px / alpha
    cost_y = py / beta
    if cost_x < cost_y:
        x_star, y_star = u_target / alpha, 0.0
    elif cost_x > cost_y:
        x_star, y_star = 0.0, u_target / beta
    else:
        x_star = u_target / (2 * alpha)
        y_star = u_target / (2 * beta)
    return DemandSolution(x_star, y_star, u_target, is_interior=False, method='analytical')


def hicksian_perfect_complements(alpha: float, beta: float,
                                  px: float, py: float, u_target: float) -> DemandSolution:
    x_star = u_target / alpha
    y_star = u_target / beta
    return DemandSolution(x_star, y_star, u_target, is_interior=True, method='analytical')


def hicksian_ces(alpha: float, rho: float,
                 px: float, py: float, u_target: float) -> DemandSolution:
    if abs(rho) < 1e-10:
        return hicksian_cobb_douglas(alpha, 1 - alpha, px, py, u_target)
    if rho >= 0.999:
        return hicksian_perfect_substitutes(alpha, 1 - alpha, px, py, u_target)
    if rho <= -50:
        return hicksian_perfect_complements(alpha, 1 - alpha, px, py, u_target)
    
    sigma = 1 / (1 - rho)
    term_x = (alpha ** sigma) * (px ** (1 - sigma))
    term_y = ((1 - alpha) ** sigma) * (py ** (1 - sigma))
    price_index = (term_x + term_y) ** (1 / (1 - sigma))
    
    x_star = (alpha ** sigma) * (px ** (-sigma)) * u_target / (price_index ** (1 - sigma))
    y_star = ((1 - alpha) ** sigma) * (py ** (-sigma)) * u_target / (price_index ** (1 - sigma))
    
    return DemandSolution(x_star, y_star, u_target, is_interior=True, method='analytical')


# =============================================================================
# NUMERICAL FALLBACK
# =============================================================================

def marshallian_numerical(utility_func: Callable[[float, float], float],
                          px: float, py: float, income: float,
                          x0: Tuple[float, float] = None) -> DemandSolution:
    if x0 is None:
        x0 = (income / (2 * px), income / (2 * py))
    
    def neg_utility(bundle):
        try:
            return -utility_func(bundle[0], bundle[1])
        except:
            return 1e10
    
    def budget(bundle):
        return income - px * bundle[0] - py * bundle[1]
    
    result = minimize(neg_utility, x0, method='SLSQP',
                      constraints={'type': 'eq', 'fun': budget},
                      bounds=[(1e-10, income / px), (1e-10, income / py)])
    
    x_star, y_star = result.x
    utility = utility_func(x_star, y_star)
    return DemandSolution(x_star, y_star, utility, 
                          is_interior=(x_star > 1e-6 and y_star > 1e-6), 
                          method='numerical')


def hicksian_numerical(utility_func: Callable[[float, float], float],
                       px: float, py: float, u_target: float,
                       x0: Tuple[float, float] = None) -> DemandSolution:
    if x0 is None:
        x0 = (1.0, 1.0)
    
    def expenditure(bundle):
        return px * bundle[0] + py * bundle[1]
    
    def utility_constraint(bundle):
        try:
            return utility_func(bundle[0], bundle[1]) - u_target
        except:
            return -1e10
    
    result = minimize(expenditure, x0, method='SLSQP',
                      constraints={'type': 'ineq', 'fun': utility_constraint},
                      bounds=[(1e-10, None), (1e-10, None)])
    
    x_star, y_star = result.x
    return DemandSolution(x_star, y_star, u_target,
                          is_interior=(x_star > 1e-6 and y_star > 1e-6),
                          method='numerical')


# =============================================================================
# REGISTRIES
# =============================================================================

MARSHALLIAN_SOLVERS = {
    'cobb-douglas': marshallian_cobb_douglas,
    'perfect-substitutes': marshallian_perfect_substitutes,
    'perfect-complements': marshallian_perfect_complements,
    'ces': marshallian_ces,
    'quasilinear': marshallian_quasilinear,
    'stone-geary': marshallian_stone_geary,
}

HICKSIAN_SOLVERS = {
    'cobb-douglas': hicksian_cobb_douglas,
    'perfect-substitutes': hicksian_perfect_substitutes,
    'perfect-complements': hicksian_perfect_complements,
    'ces': hicksian_ces,
}
