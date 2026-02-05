"""
Market: aggregation of multiple consumers.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from .agent import Consumer


class Market:
    """
    A market with multiple consumers.
    
    Aggregates individual demands into market demand.
    """
    
    def __init__(self, consumers: List[Consumer]):
        self.consumers = consumers
    
    @property
    def n_consumers(self) -> int:
        return len(self.consumers)
    
    @property
    def total_income(self) -> float:
        return sum(c.income for c in self.consumers)
    
    @property
    def total_endowment(self) -> Optional[Dict[str, float]]:
        """Aggregate endowment (for exchange economies)."""
        if not any(c.endowment for c in self.consumers):
            return None
        return {
            'X': sum(c.endowment.get('X', 0) for c in self.consumers if c.endowment),
            'Y': sum(c.endowment.get('Y', 0) for c in self.consumers if c.endowment)
        }
    
    def individual_demands(self, prices: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        return {c.name: c.demand(prices) for c in self.consumers}
    
    def aggregate_demand(self, prices: Dict[str, float]) -> Dict[str, float]:
        total = {'X': 0.0, 'Y': 0.0}
        for consumer in self.consumers:
            d = consumer.demand(prices)
            total['X'] += d['X']
            total['Y'] += d['Y']
        return total
    
    def aggregate_excess_demand(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Sum of excess demands (demand - endowment) across all consumers."""
        total = {'X': 0.0, 'Y': 0.0}
        for consumer in self.consumers:
            ed = consumer.excess_demand(prices)
            total['X'] += ed['X']
            total['Y'] += ed['Y']
        return total
    
    def demand_schedule(
        self,
        good: str = 'X',
        price_range: Tuple[float, float] = (0.5, 10),
        other_price: float = 1.0,
        n_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """Compute demand at each price level."""
        prices = np.linspace(price_range[0], price_range[1], n_points)
        
        result = {'prices': prices, 'aggregate': np.zeros(n_points)}
        for consumer in self.consumers:
            result[consumer.name] = np.zeros(n_points)
        
        for i, p in enumerate(prices):
            price_dict = {'X': p, 'Y': other_price} if good == 'X' else {'X': other_price, 'Y': p}
            
            for consumer in self.consumers:
                q = consumer.demand(price_dict)[good]
                result[consumer.name][i] = q
                result['aggregate'][i] += q
        
        return result
    
    def summary_table(self, prices: Dict[str, float]) -> str:
        individual = self.individual_demands(prices)
        agg = self.aggregate_demand(prices)
        
        lines = [
            f"Market Summary at Px={prices['X']}, Py={prices['Y']}",
            "=" * 50,
            f"{'Consumer':<15} {'Income':>10} {'X*':>10} {'Y*':>10}",
            "-" * 50,
        ]
        
        for consumer in self.consumers:
            d = individual[consumer.name]
            lines.append(f"{consumer.name:<15} {consumer.income:>10.2f} {d['X']:>10.2f} {d['Y']:>10.2f}")
        
        lines.append("-" * 50)
        lines.append(f"{'TOTAL':<15} {self.total_income:>10.2f} {agg['X']:>10.2f} {agg['Y']:>10.2f}")
        
        return '\n'.join(lines)
    
    def price_change_analysis(self, old_prices: Dict[str, float], new_prices: Dict[str, float]) -> str:
        lines = [
            f"Price Change Analysis",
            f"Old: Px={old_prices['X']}, Py={old_prices['Y']}",
            f"New: Px={new_prices['X']}, Py={new_prices['Y']}",
            "=" * 60,
            f"{'Consumer':<12} {'ΔX':>8} {'ΔY':>8} {'ΔU':>10} {'Result':>10}",
            "-" * 60,
        ]
        
        for consumer in self.consumers:
            old_d = consumer.demand(old_prices)
            new_d = consumer.demand(new_prices)
            old_u = consumer.utility_at_prices(old_prices)
            new_u = consumer.utility_at_prices(new_prices)
            
            dx = new_d['X'] - old_d['X']
            dy = new_d['Y'] - old_d['Y']
            du = new_u - old_u
            result = "Better" if du > 0.01 else ("Worse" if du < -0.01 else "Same")
            
            lines.append(f"{consumer.name:<12} {dx:>+8.2f} {dy:>+8.2f} {du:>+10.2f} {result:>10}")
        
        return '\n'.join(lines)
    
    def __repr__(self):
        names = [c.name for c in self.consumers]
        return f"Market({names})"
