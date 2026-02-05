"""
General equilibrium with production.

- Robinson Crusoe economy (single agent, production + consumption)
- Production economy (firms + consumers)
"""

import numpy as np
from scipy.optimize import minimize, brentq
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from ..consumer import Consumer
from ..production import Firm, ProductionFunction


@dataclass
class RobinsonCrusoeEquilibrium:
    """Result of Robinson Crusoe optimization."""
    labor: float          # Hours worked
    leisure: float        # Hours of leisure
    output: float         # Goods produced
    consumption: float    # Goods consumed (= output in closed economy)
    utility: float
    wage: float           # Shadow wage (marginal product of labor)
    price: float          # Normalized to 1


class RobinsonCrusoe:
    """
    Robinson Crusoe economy: single agent who is both consumer and producer.
    
    Demonstrates equivalence between:
    1. Central planner solution (maximize utility directly)
    2. Decentralized equilibrium (consumer + firm + markets)
    
    Example
    -------
    >>> from econoclasses import Utility
    >>> from econoclasses.production import ProductionFunction
    >>> from econoclasses.equilibrium import RobinsonCrusoe
    >>> 
    >>> # Preferences over consumption (C) and leisure (R)
    >>> prefs = Utility('cobb-douglas', alpha=0.5, beta=0.5)
    >>> 
    >>> # Production: output from labor
    >>> tech = ProductionFunction('cobb-douglas', alpha=0.5, A=10)
    >>> 
    >>> rc = RobinsonCrusoe(prefs, tech, total_time=24)
    >>> eq = rc.find_equilibrium()
    >>> print(f"Work {eq.labor:.1f} hours, produce {eq.output:.1f} goods")
    """
    
    def __init__(
        self,
        preferences: 'Utility',
        technology: ProductionFunction,
        total_time: float = 24,
        capital: float = 1.0
    ):
        """
        Parameters
        ----------
        preferences : Utility
            Utility over (consumption, leisure). X = consumption, Y = leisure.
        technology : ProductionFunction
            Production function Q = f(K, L). We fix K and vary L.
        total_time : float
            Total time endowment (e.g., 24 hours).
        capital : float
            Fixed capital stock.
        """
        self.preferences = preferences
        self.technology = technology
        self.total_time = total_time
        self.capital = capital
    
    def output(self, labor: float) -> float:
        """Production given labor input."""
        return self.technology.output_at(self.capital, labor)
    
    def marginal_product_labor(self, labor: float) -> float:
        """MPL = ∂Q/∂L"""
        return self.technology.mp_l_at(self.capital, labor)
    
    # =========================================================================
    # CENTRAL PLANNER SOLUTION
    # =========================================================================
    
    def find_equilibrium(self) -> RobinsonCrusoeEquilibrium:
        """
        Find optimal allocation by maximizing utility.
        
        max U(C, R) subject to:
        - C = f(K, L)  (consumption = output)
        - R = T - L    (leisure = total time - labor)
        - L ≥ 0, R ≥ 0
        """
        def neg_utility(L):
            if L <= 0 or L >= self.total_time:
                return 1e10
            C = self.output(L)
            R = self.total_time - L
            if C <= 0 or R <= 0:
                return 1e10
            try:
                return -self.preferences.utility_at(C, R)
            except:
                return 1e10
        
        # Search for optimal labor
        result = minimize(neg_utility, self.total_time / 2,
                         method='L-BFGS-B',
                         bounds=[(0.01, self.total_time - 0.01)])
        
        L_star = result.x[0]
        R_star = self.total_time - L_star
        C_star = self.output(L_star)
        U_star = self.preferences.utility_at(C_star, R_star)
        
        # Shadow wage = marginal product of labor
        w_star = self.marginal_product_labor(L_star)
        
        return RobinsonCrusoeEquilibrium(
            labor=L_star,
            leisure=R_star,
            output=C_star,
            consumption=C_star,
            utility=U_star,
            wage=w_star,
            price=1.0
        )
    
    # =========================================================================
    # DECENTRALIZED EQUILIBRIUM
    # =========================================================================
    
    def verify_decentralization(self, eq: RobinsonCrusoeEquilibrium) -> Dict:
        """
        Verify that the planner solution can be decentralized.
        
        In decentralized economy:
        - Firm maximizes profit: π = P·Q - w·L
        - Consumer maximizes utility given wage income
        - Markets clear
        """
        w = eq.wage
        P = eq.price
        
        # Firm's FOC: P × MPL = w
        firm_foc = P * self.marginal_product_labor(eq.labor)
        
        # Consumer's budget: P·C = w·L (all income spent)
        consumer_budget = P * eq.consumption
        wage_income = w * eq.labor
        
        # Consumer's FOC: MRS = w/P
        mrs = self.preferences.mrs_at(eq.consumption, eq.leisure)
        price_ratio = w / P
        
        return {
            'firm_foc_satisfied': abs(firm_foc - w) < 0.01,
            'budget_balanced': abs(consumer_budget - wage_income) < 0.01,
            'consumer_foc_satisfied': abs(mrs - price_ratio) < 0.1,
            'firm_foc_value': firm_foc,
            'mrs': mrs,
            'wage_price_ratio': price_ratio
        }
    
    def summary(self) -> str:
        eq = self.find_equilibrium()
        verify = self.verify_decentralization(eq)
        
        return f"""Robinson Crusoe Economy
=======================
Total time: {self.total_time}
Capital: {self.capital}
Technology: {self.technology.form_name}
Preferences: {self.preferences.form_name}

OPTIMAL ALLOCATION
  Labor: {eq.labor:.4f}
  Leisure: {eq.leisure:.4f}
  Output/Consumption: {eq.output:.4f}
  Utility: {eq.utility:.4f}

DECENTRALIZED PRICES
  Wage (shadow): {eq.wage:.4f}
  Price: {eq.price:.4f}

VERIFICATION
  Firm FOC (P×MPL = w): {verify['firm_foc_satisfied']}
  Consumer FOC (MRS = w/P): {verify['consumer_foc_satisfied']}
  Budget balanced: {verify['budget_balanced']}
"""


@dataclass
class ProductionEconomyEquilibrium:
    """Result of production economy equilibrium."""
    prices: Dict[str, float]      # Good prices
    wage: float                   # Wage rate
    allocations: Dict[str, Dict]  # Consumer allocations
    production: Dict[str, float]  # Firm outputs
    profits: Dict[str, float]     # Firm profits
    excess_demand: Dict[str, float]
    converged: bool


class ProductionEconomy:
    """
    General equilibrium with production.
    
    Multiple consumers, multiple firms, labor market.
    
    Example
    -------
    >>> # Two consumers, one firm
    >>> alice = Consumer("Alice", utility, income=0, endowment={'labor': 10})
    >>> bob = Consumer("Bob", utility, income=0, endowment={'labor': 10})
    >>> firm = Firm("Factory", technology, wage=1, rental=1)
    >>> 
    >>> economy = ProductionEconomy([alice, bob], [firm])
    >>> eq = economy.find_equilibrium()
    """
    
    def __init__(
        self,
        consumers: List[Consumer],
        firms: List[Firm],
        labor_endowments: Optional[Dict[str, float]] = None
    ):
        """
        Parameters
        ----------
        consumers : list of Consumer
            Consumers with preferences over goods.
        firms : list of Firm
            Firms with production technology.
        labor_endowments : dict, optional
            Labor endowment for each consumer {name: hours}.
        """
        self.consumers = consumers
        self.firms = firms
        
        # Default labor endowments
        if labor_endowments is None:
            self.labor_endowments = {c.name: 10.0 for c in consumers}
        else:
            self.labor_endowments = labor_endowments
        
        self.total_labor = sum(self.labor_endowments.values())
    
    def firm_labor_demand(self, firm: Firm, price: float, wage: float) -> float:
        """Labor demanded by firm at given output price and wage."""
        # Update firm's wage
        old_wage = firm.wage
        firm.wage = wage
        
        # Find profit-maximizing output and labor
        L = firm.labor_demand_at_price(price)
        
        firm.wage = old_wage
        return L
    
    def firm_output(self, firm: Firm, price: float, wage: float) -> float:
        """Output supplied by firm."""
        old_wage = firm.wage
        firm.wage = wage
        Q = firm.supply_at_price(price)
        firm.wage = old_wage
        return Q
    
    def consumer_demand(self, consumer: Consumer, price: float, wage: float) -> float:
        """Consumer's demand for the good."""
        # Income = wage × labor endowment + profit share
        labor = self.labor_endowments[consumer.name]
        income = wage * labor
        
        # Assume equal profit shares
        total_profit = sum(
            f.profit_at_price(price) for f in self.firms
        )
        profit_share = total_profit / len(self.consumers)
        income += profit_share
        
        # Demand at this income and price
        # Simplification: consumer demands only the produced good
        return income / price
    
    def excess_demand_good(self, price: float, wage: float) -> float:
        """Excess demand for the good."""
        demand = sum(self.consumer_demand(c, price, wage) for c in self.consumers)
        supply = sum(self.firm_output(f, price, wage) for f in self.firms)
        return demand - supply
    
    def excess_demand_labor(self, price: float, wage: float) -> float:
        """Excess demand for labor."""
        demand = sum(self.firm_labor_demand(f, price, wage) for f in self.firms)
        supply = self.total_labor
        return demand - supply
    
    def find_equilibrium(self, price_bounds: Tuple[float, float] = (0.1, 100),
                         wage_bounds: Tuple[float, float] = (0.1, 100)) -> ProductionEconomyEquilibrium:
        """
        Find competitive equilibrium.
        
        Normalize good price P = 1, find wage w that clears labor market.
        """
        P = 1.0  # Numeraire
        
        # Find wage that clears labor market
        def labor_excess(w):
            return self.excess_demand_labor(P, w)
        
        try:
            # Check signs at bounds
            low_excess = labor_excess(wage_bounds[0])
            high_excess = labor_excess(wage_bounds[1])
            
            if low_excess * high_excess > 0:
                # Try to find sign change
                test_wages = np.linspace(wage_bounds[0], wage_bounds[1], 50)
                for i in range(len(test_wages) - 1):
                    if labor_excess(test_wages[i]) * labor_excess(test_wages[i+1]) < 0:
                        wage_bounds = (test_wages[i], test_wages[i+1])
                        break
            
            w_star = brentq(labor_excess, wage_bounds[0], wage_bounds[1])
            converged = True
        except:
            w_star = 1.0
            converged = False
        
        # Compute equilibrium quantities
        allocations = {}
        for c in self.consumers:
            demand = self.consumer_demand(c, P, w_star)
            allocations[c.name] = {'good': demand, 'labor_supplied': self.labor_endowments[c.name]}
        
        production = {}
        profits = {}
        for f in self.firms:
            production[f.name] = self.firm_output(f, P, w_star)
            old_wage = f.wage
            f.wage = w_star
            profits[f.name] = f.profit_at_price(P)
            f.wage = old_wage
        
        return ProductionEconomyEquilibrium(
            prices={'good': P},
            wage=w_star,
            allocations=allocations,
            production=production,
            profits=profits,
            excess_demand={
                'good': self.excess_demand_good(P, w_star),
                'labor': self.excess_demand_labor(P, w_star)
            },
            converged=converged
        )
    
    def summary(self) -> str:
        eq = self.find_equilibrium()
        
        lines = [
            "Production Economy Equilibrium",
            "=" * 50,
            f"Consumers: {[c.name for c in self.consumers]}",
            f"Firms: {[f.name for f in self.firms]}",
            f"Total labor supply: {self.total_labor}",
            "",
            "PRICES",
            f"  Good price: {eq.prices['good']:.4f}",
            f"  Wage: {eq.wage:.4f}",
            "",
            "PRODUCTION",
        ]
        
        for name, Q in eq.production.items():
            profit = eq.profits[name]
            lines.append(f"  {name}: Q={Q:.4f}, π={profit:.4f}")
        
        lines.append("")
        lines.append("CONSUMPTION")
        
        for name, alloc in eq.allocations.items():
            lines.append(f"  {name}: C={alloc['good']:.4f}")
        
        lines.append("")
        lines.append(f"Excess demand (good): {eq.excess_demand['good']:.6f}")
        lines.append(f"Excess demand (labor): {eq.excess_demand['labor']:.6f}")
        lines.append(f"Converged: {eq.converged}")
        
        return '\n'.join(lines)
