"""
Firm class: production technology + factor prices.

Generates cost curves and supply functions.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from .technology import ProductionFunction


@dataclass
class SupplyPoint:
    """A point on the supply curve."""
    price: float
    quantity: float
    profit: float


class Firm:
    """
    A firm with technology and factor prices.
    
    Combines ProductionFunction with wage and rental rate to compute
    cost curves and supply.
    
    Example
    -------
    >>> tech = ProductionFunction('cobb-douglas', alpha=0.3, A=2)
    >>> firm = Firm("Acme", tech, wage=5, rental=10)
    >>> print(firm.cost_function(100))
    >>> print(firm.supply_at_price(15))
    """
    
    def __init__(self, name: str, technology: ProductionFunction,
                 wage: float, rental: float):
        self.name = name
        self.technology = technology
        self.wage = wage
        self.rental = rental
    
    # =========================================================================
    # COST FUNCTIONS
    # =========================================================================
    
    def total_cost(self, Q: float) -> float:
        """C(Q) = minimum cost to produce Q."""
        if Q <= 0:
            return 0
        return self.technology.cost_function(Q, self.wage, self.rental)
    
    def average_cost(self, Q: float) -> float:
        """AC(Q) = C(Q) / Q."""
        if Q <= 0:
            return float('inf')
        return self.total_cost(Q) / Q
    
    def marginal_cost(self, Q: float, delta: float = 0.01) -> float:
        """MC(Q) = ∂C/∂Q."""
        if Q <= 0:
            return self.marginal_cost(0.01, delta)
        return self.technology.marginal_cost(Q, self.wage, self.rental, delta)
    
    def average_variable_cost(self, Q: float, fixed_cost: float = 0) -> float:
        """AVC(Q) = (C(Q) - FC) / Q."""
        if Q <= 0:
            return float('inf')
        return (self.total_cost(Q) - fixed_cost) / Q
    
    # =========================================================================
    # COST CURVES (for plotting)
    # =========================================================================
    
    def cost_curves(self, Q_range: Tuple[float, float], n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute cost curves over a range of output.
        
        Returns dict with 'Q', 'TC', 'AC', 'MC' arrays.
        """
        Q_vals = np.linspace(max(Q_range[0], 0.1), Q_range[1], n_points)
        
        TC = np.array([self.total_cost(q) for q in Q_vals])
        AC = TC / Q_vals
        MC = np.array([self.marginal_cost(q) for q in Q_vals])
        
        return {
            'Q': Q_vals,
            'TC': TC,
            'AC': AC,
            'MC': MC
        }
    
    # =========================================================================
    # SUPPLY
    # =========================================================================
    
    def supply_at_price(self, price: float) -> float:
        """
        Quantity supplied at given price.
        
        Q* such that MC(Q*) = P, if P ≥ min AVC.
        """
        # Find Q where MC = P using binary search
        Q_low, Q_high = 0.01, 1000
        
        for _ in range(50):
            Q_mid = (Q_low + Q_high) / 2
            mc = self.marginal_cost(Q_mid)
            
            if mc < price:
                Q_low = Q_mid
            else:
                Q_high = Q_mid
            
            if abs(mc - price) < 0.01:
                break
        
        # Check shutdown condition (P < min AVC)
        ac = self.average_cost(Q_mid)
        if price < ac * 0.8:  # Rough shutdown check
            return 0
        
        return Q_mid
    
    def profit_at_price(self, price: float) -> float:
        """Profit when selling at given price."""
        Q = self.supply_at_price(price)
        if Q <= 0:
            return 0
        return price * Q - self.total_cost(Q)
    
    def supply_curve(self, price_range: Tuple[float, float], 
                     n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute supply curve over a price range.
        
        Returns dict with 'P' and 'Q' arrays.
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        quantities = np.array([self.supply_at_price(p) for p in prices])
        profits = np.array([self.profit_at_price(p) for p in prices])
        
        return {
            'P': prices,
            'Q': quantities,
            'profit': profits
        }
    
    # =========================================================================
    # INPUT DEMANDS
    # =========================================================================
    
    def conditional_input_demand(self, Q: float) -> Dict[str, float]:
        """
        Cost-minimizing input bundle for output Q.
        """
        solution = self.technology.cost_minimize(Q, self.wage, self.rental)
        return {'K': solution.K, 'L': solution.L}
    
    def labor_demand_at_price(self, price: float) -> float:
        """Labor demand when output price is P."""
        Q = self.supply_at_price(price)
        if Q <= 0:
            return 0
        return self.conditional_input_demand(Q)['L']
    
    def capital_demand_at_price(self, price: float) -> float:
        """Capital demand when output price is P."""
        Q = self.supply_at_price(price)
        if Q <= 0:
            return 0
        return self.conditional_input_demand(Q)['K']
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def __repr__(self):
        return f"Firm('{self.name}', {self.technology.form_name}, w={self.wage}, r={self.rental})"
    
    def summary(self, price: float = 10) -> str:
        Q = self.supply_at_price(price)
        inputs = self.conditional_input_demand(Q) if Q > 0 else {'K': 0, 'L': 0}
        profit = self.profit_at_price(price)
        
        return f"""Firm: {self.name}
Technology: {self.technology.form_name}
Factor prices: w={self.wage}, r={self.rental}

At output price P={price}:
  Quantity supplied: Q*={Q:.4f}
  Inputs: K*={inputs['K']:.4f}, L*={inputs['L']:.4f}
  Revenue: {price * Q:.4f}
  Cost: {self.total_cost(Q):.4f}
  Profit: {profit:.4f}"""


class Industry:
    """
    An industry: collection of firms.
    
    Aggregates individual supply curves into market supply.
    """
    
    def __init__(self, firms: List[Firm]):
        self.firms = firms
    
    @property
    def n_firms(self) -> int:
        return len(self.firms)
    
    def aggregate_supply(self, price: float) -> float:
        """Total quantity supplied at price."""
        return sum(f.supply_at_price(price) for f in self.firms)
    
    def supply_curve(self, price_range: Tuple[float, float],
                     n_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Market supply curve.
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        
        result = {'P': prices, 'aggregate': np.zeros(n_points)}
        for firm in self.firms:
            result[firm.name] = np.zeros(n_points)
        
        for i, p in enumerate(prices):
            for firm in self.firms:
                q = firm.supply_at_price(p)
                result[firm.name][i] = q
                result['aggregate'][i] += q
        
        return result
    
    def summary_table(self, price: float) -> str:
        """Formatted table of firm supplies."""
        lines = [
            f"Industry Supply at P={price}",
            "=" * 50,
            f"{'Firm':<15} {'Q':>10} {'Profit':>10}",
            "-" * 50,
        ]
        
        total_q = 0
        total_profit = 0
        
        for firm in self.firms:
            q = firm.supply_at_price(price)
            profit = firm.profit_at_price(price)
            total_q += q
            total_profit += profit
            lines.append(f"{firm.name:<15} {q:>10.2f} {profit:>10.2f}")
        
        lines.append("-" * 50)
        lines.append(f"{'TOTAL':<15} {total_q:>10.2f} {total_profit:>10.2f}")
        
        return '\n'.join(lines)
    
    def __repr__(self):
        names = [f.name for f in self.firms]
        return f"Industry({names})"
