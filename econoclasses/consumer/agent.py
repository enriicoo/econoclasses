"""
Individual consumer agent.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..preferences import Utility


@dataclass
class Consumer:
    """
    A consumer with preferences (utility function), income, and optional endowment.
    
    For exchange economies, endowment specifies initial holdings.
    """
    name: str
    utility: Utility
    income: float
    endowment: Optional[Dict[str, float]] = None  # {'X': qty, 'Y': qty}
    
    def __post_init__(self):
        self.utility.income = self.income
        
        # If endowment specified, income = value of endowment at current prices
        if self.endowment is not None:
            self._update_income_from_endowment()
    
    def _update_income_from_endowment(self):
        """Compute income as value of endowment at current prices."""
        if self.endowment is not None:
            self.income = (self.endowment.get('X', 0) * self.utility.price_x +
                          self.endowment.get('Y', 0) * self.utility.price_y)
            self.utility.income = self.income
    
    def set_prices(self, price_x: float, price_y: float):
        """Update prices (and recompute income if endowment-based)."""
        self.utility.price_x = price_x
        self.utility.price_y = price_y
        if self.endowment is not None:
            self._update_income_from_endowment()
    
    def demand(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Compute optimal bundle at given prices."""
        if self.endowment is not None:
            income = (self.endowment.get('X', 0) * prices['X'] +
                     self.endowment.get('Y', 0) * prices['Y'])
        else:
            income = self.income
        
        x, y = self.utility.demand_at_prices(
            price_x=prices['X'],
            price_y=prices['Y'],
            income=income
        )
        return {'X': x, 'Y': y}
    
    def excess_demand(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Demand minus endowment: what the consumer wants to buy (positive) or sell (negative).
        
        Only meaningful for exchange economies with endowments.
        """
        d = self.demand(prices)
        if self.endowment is None:
            return d
        return {
            'X': d['X'] - self.endowment.get('X', 0),
            'Y': d['Y'] - self.endowment.get('Y', 0)
        }
    
    def utility_at_prices(self, prices: Dict[str, float]) -> float:
        bundle = self.demand(prices)
        return self.utility.utility_at(bundle['X'], bundle['Y'])
    
    def utility_at_bundle(self, bundle: Dict[str, float]) -> float:
        return self.utility.utility_at(bundle['X'], bundle['Y'])
    
    def __repr__(self):
        if self.endowment:
            return f"Consumer('{self.name}', {self.utility.form_name}, ω={self.endowment})"
        return f"Consumer('{self.name}', {self.utility.form_name}, I={self.income})"
