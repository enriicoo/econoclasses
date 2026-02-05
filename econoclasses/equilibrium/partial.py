"""
Partial equilibrium analysis.

Single-market supply and demand analysis.
Note: Full implementation requires production module (future work).
"""

import numpy as np
from scipy.optimize import brentq
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PartialEquilibriumResult:
    """Result of partial equilibrium computation."""
    price: float
    quantity: float
    consumer_surplus: float
    producer_surplus: float
    total_surplus: float


class PartialEquilibrium:
    """
    Single-market partial equilibrium.
    
    Supply and demand curves are specified as functions of price.
    
    Example
    -------
    >>> # Linear demand: Q_d = 100 - 2P
    >>> # Linear supply: Q_s = -20 + 3P
    >>> pe = PartialEquilibrium(
    ...     demand_func=lambda p: 100 - 2*p,
    ...     supply_func=lambda p: -20 + 3*p,
    ...     price_bounds=(0, 50)
    ... )
    >>> eq = pe.find_equilibrium()
    >>> print(f"P*={eq.price:.2f}, Q*={eq.quantity:.2f}")
    """
    
    def __init__(
        self,
        demand_func: Callable[[float], float],
        supply_func: Callable[[float], float],
        price_bounds: Tuple[float, float] = (0.01, 100),
        demand_intercept: Optional[float] = None,  # Max price consumers pay
        supply_intercept: Optional[float] = None,  # Min price suppliers accept
    ):
        """
        Initialize partial equilibrium model.
        
        Parameters
        ----------
        demand_func : callable
            Q_d(P) - quantity demanded as function of price
        supply_func : callable
            Q_s(P) - quantity supplied as function of price
        price_bounds : tuple
            Range for equilibrium price search
        demand_intercept : float, optional
            Price where demand = 0 (for surplus calculation)
        supply_intercept : float, optional
            Price where supply = 0 (for surplus calculation)
        """
        self.demand = demand_func
        self.supply = supply_func
        self.price_bounds = price_bounds
        
        # Try to infer intercepts if not provided
        self.demand_intercept = demand_intercept
        self.supply_intercept = supply_intercept
    
    def excess_demand(self, price: float) -> float:
        """Q_d(P) - Q_s(P)"""
        return self.demand(price) - self.supply(price)
    
    def find_equilibrium(self) -> PartialEquilibriumResult:
        """Find market-clearing price and quantity."""
        
        # Find price where excess demand = 0
        try:
            p_star = brentq(self.excess_demand, 
                           self.price_bounds[0], 
                           self.price_bounds[1])
        except ValueError:
            # No equilibrium in range
            return PartialEquilibriumResult(
                price=np.nan, quantity=np.nan,
                consumer_surplus=np.nan, producer_surplus=np.nan,
                total_surplus=np.nan
            )
        
        q_star = self.demand(p_star)
        
        # Compute surpluses
        cs = self._consumer_surplus(p_star, q_star)
        ps = self._producer_surplus(p_star, q_star)
        
        return PartialEquilibriumResult(
            price=p_star,
            quantity=q_star,
            consumer_surplus=cs,
            producer_surplus=ps,
            total_surplus=cs + ps
        )
    
    def _consumer_surplus(self, p_eq: float, q_eq: float) -> float:
        """
        Area under demand curve above equilibrium price.
        
        CS = ∫[0 to q_eq] P_d(q) dq - p_eq * q_eq
        
        For numerical integration, we use the inverse demand if available,
        otherwise approximate.
        """
        if self.demand_intercept is not None:
            # Assuming linear demand for simplicity
            # CS = 0.5 * (P_max - P_eq) * Q_eq
            return 0.5 * (self.demand_intercept - p_eq) * q_eq
        else:
            # Numerical approximation
            n_points = 100
            prices = np.linspace(p_eq, self.price_bounds[1], n_points)
            quantities = [max(0, self.demand(p)) for p in prices]
            # Approximate integral
            return np.trapz(quantities, prices)
    
    def _producer_surplus(self, p_eq: float, q_eq: float) -> float:
        """
        Area above supply curve below equilibrium price.
        
        PS = p_eq * q_eq - ∫[0 to q_eq] P_s(q) dq
        """
        if self.supply_intercept is not None:
            # Assuming linear supply
            # PS = 0.5 * (P_eq - P_min) * Q_eq
            return 0.5 * (p_eq - self.supply_intercept) * q_eq
        else:
            # Numerical approximation
            n_points = 100
            prices = np.linspace(self.price_bounds[0], p_eq, n_points)
            quantities = [max(0, self.supply(p)) for p in prices]
            return p_eq * q_eq - np.trapz(quantities, prices)
    
    def analyze_tax(self, tax: float, on_consumer: bool = True) -> Dict:
        """
        Analyze the effect of a per-unit tax.
        
        Parameters
        ----------
        tax : float
            Per-unit tax amount
        on_consumer : bool
            If True, consumers pay price + tax. If False, suppliers receive price - tax.
        
        Returns
        -------
        dict with before/after equilibrium and deadweight loss
        """
        # Pre-tax equilibrium
        pre = self.find_equilibrium()
        
        # Post-tax: shift the relevant curve
        if on_consumer:
            # Demand shifts down by tax amount
            taxed_demand = lambda p: self.demand(p + tax)
            post_eq = PartialEquilibrium(
                taxed_demand, self.supply, self.price_bounds
            ).find_equilibrium()
            price_consumers_pay = post_eq.price + tax
            price_producers_receive = post_eq.price
        else:
            # Supply shifts up by tax amount
            taxed_supply = lambda p: self.supply(p - tax)
            post_eq = PartialEquilibrium(
                self.demand, taxed_supply, self.price_bounds
            ).find_equilibrium()
            price_consumers_pay = post_eq.price
            price_producers_receive = post_eq.price - tax
        
        # Tax revenue
        tax_revenue = tax * post_eq.quantity
        
        # Deadweight loss = lost surplus - tax revenue
        dwl = (pre.total_surplus - post_eq.total_surplus) - tax_revenue
        
        return {
            'pre_tax': pre,
            'post_tax': post_eq,
            'price_consumers': price_consumers_pay,
            'price_producers': price_producers_receive,
            'tax_revenue': tax_revenue,
            'deadweight_loss': max(0, dwl),
            'quantity_reduction': pre.quantity - post_eq.quantity
        }
    
    def summary(self) -> str:
        eq = self.find_equilibrium()
        return f"""Partial Equilibrium Analysis
============================
Equilibrium price: {eq.price:.4f}
Equilibrium quantity: {eq.quantity:.4f}

Consumer surplus: {eq.consumer_surplus:.4f}
Producer surplus: {eq.producer_surplus:.4f}
Total surplus: {eq.total_surplus:.4f}
"""


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON CASES
# =============================================================================

def linear_demand(intercept: float, slope: float) -> Callable[[float], float]:
    """
    Create linear demand function: Q = intercept - slope * P
    
    Example: linear_demand(100, 2) gives Q = 100 - 2P
    """
    return lambda p: intercept - slope * p


def linear_supply(intercept: float, slope: float) -> Callable[[float], float]:
    """
    Create linear supply function: Q = intercept + slope * P
    
    Example: linear_supply(-20, 3) gives Q = -20 + 3P
    """
    return lambda p: intercept + slope * p


def constant_elasticity_demand(a: float, elasticity: float) -> Callable[[float], float]:
    """
    Create constant elasticity demand: Q = a * P^elasticity
    
    For normal demand, elasticity < 0.
    """
    return lambda p: a * (p ** elasticity)
