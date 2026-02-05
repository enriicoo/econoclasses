"""
Exchange economy equilibrium.

Two-consumer pure exchange economy:
- Edgeworth box visualization
- Contract curve (Pareto efficient allocations)
- Walrasian equilibrium (market-clearing prices)
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from ..consumer import Consumer


@dataclass
class Allocation:
    """An allocation of goods between two consumers."""
    consumer_a: Dict[str, float]  # {'X': qty, 'Y': qty}
    consumer_b: Dict[str, float]
    
    def is_feasible(self, total_x: float, total_y: float, tol: float = 1e-6) -> bool:
        """Check if allocation uses exactly the total endowment."""
        return (abs(self.consumer_a['X'] + self.consumer_b['X'] - total_x) < tol and
                abs(self.consumer_a['Y'] + self.consumer_b['Y'] - total_y) < tol)


@dataclass 
class EquilibriumResult:
    """Result of equilibrium computation."""
    prices: Dict[str, float]  # Equilibrium prices
    allocation: Allocation    # Equilibrium allocation
    excess_demand: Dict[str, float]  # Should be ~0 at equilibrium
    converged: bool
    iterations: int


class ExchangeEconomy:
    """
    A two-consumer pure exchange economy.
    
    Each consumer has:
    - Utility function (preferences)
    - Endowment (initial bundle)
    
    Key concepts:
    - Edgeworth box: all feasible allocations
    - Contract curve: Pareto efficient allocations (MRS_A = MRS_B)
    - Core: allocations both prefer to endowment
    - Walrasian equilibrium: prices where excess demand = 0
    
    Example
    -------
    >>> from econoclasses import Utility, Consumer
    >>> from econoclasses.equilibrium import ExchangeEconomy
    >>> 
    >>> alice = Consumer("Alice", 
    ...     Utility('cobb-douglas', alpha=0.5, beta=0.5),
    ...     income=0,  # Will be computed from endowment
    ...     endowment={'X': 10, 'Y': 2})
    >>> 
    >>> bob = Consumer("Bob",
    ...     Utility('cobb-douglas', alpha=0.3, beta=0.7),
    ...     income=0,
    ...     endowment={'X': 2, 'Y': 10})
    >>> 
    >>> economy = ExchangeEconomy(alice, bob)
    >>> eq = economy.find_equilibrium()
    >>> print(eq.prices)
    """
    
    def __init__(self, consumer_a: Consumer, consumer_b: Consumer):
        """
        Create exchange economy with two consumers.
        
        Both consumers must have endowments specified.
        """
        if consumer_a.endowment is None or consumer_b.endowment is None:
            raise ValueError("Both consumers must have endowments for exchange economy")
        
        self.consumer_a = consumer_a
        self.consumer_b = consumer_b
        
        # Total resources in the economy
        self.total_x = consumer_a.endowment['X'] + consumer_b.endowment['X']
        self.total_y = consumer_a.endowment['Y'] + consumer_b.endowment['Y']
        
        # Initial endowment point
        self.endowment = Allocation(
            consumer_a=dict(consumer_a.endowment),
            consumer_b=dict(consumer_b.endowment)
        )
    
    @property
    def endowment_utilities(self) -> Tuple[float, float]:
        """Utility of each consumer at their endowment."""
        u_a = self.consumer_a.utility.utility_at(
            self.endowment.consumer_a['X'], 
            self.endowment.consumer_a['Y']
        )
        u_b = self.consumer_b.utility.utility_at(
            self.endowment.consumer_b['X'],
            self.endowment.consumer_b['Y']
        )
        return u_a, u_b
    
    # =========================================================================
    # EXCESS DEMAND
    # =========================================================================
    
    def excess_demand(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate excess demand at given prices.
        
        Z(p) = Σ (demand_i - endowment_i)
        
        At equilibrium, Z(p) = 0 for all goods.
        """
        ed_a = self.consumer_a.excess_demand(prices)
        ed_b = self.consumer_b.excess_demand(prices)
        return {
            'X': ed_a['X'] + ed_b['X'],
            'Y': ed_a['Y'] + ed_b['Y']
        }
    
    def excess_demand_x(self, price_ratio: float) -> float:
        """
        Excess demand for X as function of Px/Py.
        
        We normalize Py = 1, so price_ratio = Px.
        By Walras' Law, if Z_x = 0 then Z_y = 0.
        """
        prices = {'X': price_ratio, 'Y': 1.0}
        return self.excess_demand(prices)['X']
    
    # =========================================================================
    # WALRASIAN EQUILIBRIUM
    # =========================================================================
    
    def find_equilibrium(self, 
                         price_bounds: Tuple[float, float] = (0.01, 100),
                         normalize_y: bool = True) -> EquilibriumResult:
        """
        Find Walrasian (competitive) equilibrium prices.
        
        Uses Walras' Law: we only need to clear one market.
        Normalizes Py = 1 and finds Px that clears market for X.
        
        Parameters
        ----------
        price_bounds : tuple
            Range to search for equilibrium price ratio.
        normalize_y : bool
            If True, set Py=1 and find Px. If False, set Px=1.
        
        Returns
        -------
        EquilibriumResult with prices, allocation, and convergence info.
        """
        # Check that excess demand changes sign in the bounds
        z_low = self.excess_demand_x(price_bounds[0])
        z_high = self.excess_demand_x(price_bounds[1])
        
        # If same sign, try to find better bounds
        if z_low * z_high > 0:
            # Search for sign change
            test_prices = np.logspace(np.log10(price_bounds[0]), 
                                       np.log10(price_bounds[1]), 50)
            z_values = [self.excess_demand_x(p) for p in test_prices]
            
            # Find sign change
            for i in range(len(z_values) - 1):
                if z_values[i] * z_values[i+1] < 0:
                    price_bounds = (test_prices[i], test_prices[i+1])
                    break
            else:
                # No equilibrium found in range
                return EquilibriumResult(
                    prices={'X': np.nan, 'Y': 1.0},
                    allocation=self.endowment,
                    excess_demand={'X': np.nan, 'Y': np.nan},
                    converged=False,
                    iterations=0
                )
        
        # Use Brent's method to find root
        try:
            px_star, result = brentq(self.excess_demand_x, 
                                      price_bounds[0], price_bounds[1],
                                      full_output=True)
            converged = result.converged
            iterations = result.iterations
        except ValueError:
            return EquilibriumResult(
                prices={'X': np.nan, 'Y': 1.0},
                allocation=self.endowment,
                excess_demand={'X': np.nan, 'Y': np.nan},
                converged=False,
                iterations=0
            )
        
        # Compute equilibrium allocation
        prices = {'X': px_star, 'Y': 1.0}
        alloc_a = self.consumer_a.demand(prices)
        alloc_b = self.consumer_b.demand(prices)
        
        allocation = Allocation(
            consumer_a=alloc_a,
            consumer_b=alloc_b
        )
        
        excess = self.excess_demand(prices)
        
        return EquilibriumResult(
            prices=prices,
            allocation=allocation,
            excess_demand=excess,
            converged=converged,
            iterations=iterations
        )
    
    # =========================================================================
    # CONTRACT CURVE (PARETO EFFICIENCY)
    # =========================================================================
    
    def is_pareto_efficient(self, allocation: Allocation, tol: float = 1e-4) -> bool:
        """
        Check if allocation is Pareto efficient.
        
        At Pareto efficiency: MRS_A = MRS_B
        """
        mrs_a = self.consumer_a.utility.mrs_at(
            allocation.consumer_a['X'], allocation.consumer_a['Y']
        )
        mrs_b = self.consumer_b.utility.mrs_at(
            allocation.consumer_b['X'], allocation.consumer_b['Y']
        )
        return abs(mrs_a - mrs_b) < tol
    
    def contract_curve(self, n_points: int = 50) -> List[Allocation]:
        """
        Compute points on the contract curve.
        
        The contract curve is the set of Pareto efficient allocations.
        For each level of X_A, find Y_A where MRS_A = MRS_B.
        
        Returns list of Pareto efficient allocations.
        """
        contract_points = []
        
        x_values = np.linspace(0.01, self.total_x - 0.01, n_points)
        
        for x_a in x_values:
            x_b = self.total_x - x_a
            
            # Find y_a where MRS_A(x_a, y_a) = MRS_B(x_b, total_y - y_a)
            def mrs_diff(y_a):
                if y_a <= 0.01 or y_a >= self.total_y - 0.01:
                    return 1e10
                y_b = self.total_y - y_a
                try:
                    mrs_a = self.consumer_a.utility.mrs_at(x_a, y_a)
                    mrs_b = self.consumer_b.utility.mrs_at(x_b, y_b)
                    return (mrs_a - mrs_b) ** 2
                except:
                    return 1e10
            
            # Minimize squared MRS difference
            result = minimize_scalar(mrs_diff, bounds=(0.01, self.total_y - 0.01), method='bounded')
            
            if result.fun < 1e-6:  # Found efficient point
                y_a = result.x
                y_b = self.total_y - y_a
                contract_points.append(Allocation(
                    consumer_a={'X': x_a, 'Y': y_a},
                    consumer_b={'X': x_b, 'Y': y_b}
                ))
        
        return contract_points
    
    def contract_curve_arrays(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return contract curve as arrays for plotting.
        
        Returns (x_a_array, y_a_array) - coordinates in consumer A's frame.
        """
        points = self.contract_curve(n_points)
        x_a = np.array([p.consumer_a['X'] for p in points])
        y_a = np.array([p.consumer_a['Y'] for p in points])
        return x_a, y_a
    
    # =========================================================================
    # CORE
    # =========================================================================
    
    def is_in_core(self, allocation: Allocation) -> bool:
        """
        Check if allocation is in the core.
        
        An allocation is in the core if:
        1. It's Pareto efficient
        2. Both consumers are at least as well off as at endowment
        """
        # Check individual rationality
        u_a_endow, u_b_endow = self.endowment_utilities
        
        u_a = self.consumer_a.utility.utility_at(
            allocation.consumer_a['X'], allocation.consumer_a['Y']
        )
        u_b = self.consumer_b.utility.utility_at(
            allocation.consumer_b['X'], allocation.consumer_b['Y']
        )
        
        if u_a < u_a_endow - 1e-6 or u_b < u_b_endow - 1e-6:
            return False
        
        return self.is_pareto_efficient(allocation)
    
    def core_bounds(self) -> Tuple[float, float]:
        """
        Find X_A bounds for allocations in the core.
        
        Returns (x_a_min, x_a_max) on the contract curve that are in the core.
        """
        contract = self.contract_curve(100)
        core_points = [p for p in contract if self.is_in_core(p)]
        
        if not core_points:
            return (np.nan, np.nan)
        
        x_values = [p.consumer_a['X'] for p in core_points]
        return (min(x_values), max(x_values))
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def gains_from_trade(self, allocation: Optional[Allocation] = None) -> Dict[str, float]:
        """
        Compute utility gain for each consumer relative to endowment.
        
        If no allocation given, uses equilibrium allocation.
        """
        if allocation is None:
            eq = self.find_equilibrium()
            allocation = eq.allocation
        
        u_a_endow, u_b_endow = self.endowment_utilities
        
        u_a = self.consumer_a.utility.utility_at(
            allocation.consumer_a['X'], allocation.consumer_a['Y']
        )
        u_b = self.consumer_b.utility.utility_at(
            allocation.consumer_b['X'], allocation.consumer_b['Y']
        )
        
        return {
            self.consumer_a.name: u_a - u_a_endow,
            self.consumer_b.name: u_b - u_b_endow,
            'total': (u_a - u_a_endow) + (u_b - u_b_endow)
        }
    
    def summary(self) -> str:
        """Print summary of the exchange economy."""
        eq = self.find_equilibrium()
        gains = self.gains_from_trade(eq.allocation)
        u_a_endow, u_b_endow = self.endowment_utilities
        
        lines = [
            "=" * 60,
            "EXCHANGE ECONOMY SUMMARY",
            "=" * 60,
            "",
            f"Consumer A: {self.consumer_a.name}",
            f"  Utility: {self.consumer_a.utility.form_name}",
            f"  Endowment: X={self.consumer_a.endowment['X']}, Y={self.consumer_a.endowment['Y']}",
            f"  Utility at endowment: {u_a_endow:.4f}",
            "",
            f"Consumer B: {self.consumer_b.name}",
            f"  Utility: {self.consumer_b.utility.form_name}",
            f"  Endowment: X={self.consumer_b.endowment['X']}, Y={self.consumer_b.endowment['Y']}",
            f"  Utility at endowment: {u_b_endow:.4f}",
            "",
            f"Total resources: X={self.total_x}, Y={self.total_y}",
            "",
            "-" * 60,
            "WALRASIAN EQUILIBRIUM",
            "-" * 60,
            f"Equilibrium prices: Px={eq.prices['X']:.4f}, Py={eq.prices['Y']:.4f}",
            f"Price ratio Px/Py: {eq.prices['X']/eq.prices['Y']:.4f}",
            "",
            f"{self.consumer_a.name}'s allocation: X={eq.allocation.consumer_a['X']:.4f}, Y={eq.allocation.consumer_a['Y']:.4f}",
            f"{self.consumer_b.name}'s allocation: X={eq.allocation.consumer_b['X']:.4f}, Y={eq.allocation.consumer_b['Y']:.4f}",
            "",
            f"Excess demand: X={eq.excess_demand['X']:.6f}, Y={eq.excess_demand['Y']:.6f}",
            f"Converged: {eq.converged}",
            "",
            "-" * 60,
            "GAINS FROM TRADE",
            "-" * 60,
            f"{self.consumer_a.name}: {gains[self.consumer_a.name]:+.4f}",
            f"{self.consumer_b.name}: {gains[self.consumer_b.name]:+.4f}",
            f"Total welfare gain: {gains['total']:+.4f}",
            "",
            f"Is equilibrium Pareto efficient? {self.is_pareto_efficient(eq.allocation)}",
            f"Is equilibrium in core? {self.is_in_core(eq.allocation)}",
        ]
        
        return '\n'.join(lines)
    
    def __repr__(self):
        return f"ExchangeEconomy({self.consumer_a.name}, {self.consumer_b.name})"
