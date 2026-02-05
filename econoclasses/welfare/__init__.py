"""
Welfare economics: efficiency, equity, and social welfare.

- Pareto efficiency criteria
- Welfare theorems
- Social welfare functions
- Inequality measures
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass

from ..consumer import Consumer, Market
from ..equilibrium.exchange import ExchangeEconomy, Allocation


# =============================================================================
# PARETO EFFICIENCY
# =============================================================================

def is_pareto_improvement(
    economy: ExchangeEconomy,
    old_alloc: Allocation,
    new_alloc: Allocation
) -> bool:
    """
    Check if new allocation is a Pareto improvement over old.
    
    Pareto improvement: at least one person better off, no one worse off.
    """
    u_a_old = economy.consumer_a.utility.utility_at(
        old_alloc.consumer_a['X'], old_alloc.consumer_a['Y'])
    u_b_old = economy.consumer_b.utility.utility_at(
        old_alloc.consumer_b['X'], old_alloc.consumer_b['Y'])
    
    u_a_new = economy.consumer_a.utility.utility_at(
        new_alloc.consumer_a['X'], new_alloc.consumer_a['Y'])
    u_b_new = economy.consumer_b.utility.utility_at(
        new_alloc.consumer_b['X'], new_alloc.consumer_b['Y'])
    
    # At least one strictly better, none worse
    someone_better = (u_a_new > u_a_old + 1e-6) or (u_b_new > u_b_old + 1e-6)
    no_one_worse = (u_a_new >= u_a_old - 1e-6) and (u_b_new >= u_b_old - 1e-6)
    
    return someone_better and no_one_worse


def pareto_dominates(
    economy: ExchangeEconomy,
    alloc1: Allocation,
    alloc2: Allocation
) -> bool:
    """Check if alloc1 Pareto dominates alloc2."""
    return is_pareto_improvement(economy, alloc2, alloc1)


# =============================================================================
# SOCIAL WELFARE FUNCTIONS
# =============================================================================

def utilitarian_welfare(utilities: List[float]) -> float:
    """
    Benthamite/Utilitarian: W = Σ Ui
    
    Maximizes total utility. Allows tradeoffs between people.
    """
    return sum(utilities)


def rawlsian_welfare(utilities: List[float]) -> float:
    """
    Rawlsian/Maximin: W = min(Ui)
    
    Maximizes welfare of worst-off person. Extreme inequality aversion.
    """
    return min(utilities)


def nash_welfare(utilities: List[float]) -> float:
    """
    Nash/Geometric: W = Π Ui  (or equivalently Σ log(Ui))
    
    Balances efficiency and equality. Scale-invariant.
    """
    return np.prod(utilities)


def cobb_douglas_welfare(utilities: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Generalized: W = Π Ui^αi
    
    With equal weights, equivalent to Nash.
    """
    if weights is None:
        weights = [1.0 / len(utilities)] * len(utilities)
    return np.prod([u ** w for u, w in zip(utilities, weights)])


def atkinson_welfare(utilities: List[float], epsilon: float = 0.5) -> float:
    """
    Atkinson: W = [Σ Ui^(1-ε)]^(1/(1-ε))  for ε ≠ 1
             W = Π Ui^(1/n)               for ε = 1
    
    ε = 0: Utilitarian
    ε → ∞: Rawlsian
    """
    n = len(utilities)
    if abs(epsilon - 1) < 1e-6:
        return np.prod(utilities) ** (1/n)
    else:
        return (sum(u ** (1 - epsilon) for u in utilities) / n) ** (1 / (1 - epsilon))


@dataclass
class WelfareAnalysis:
    """Results of welfare analysis for an allocation."""
    utilities: Dict[str, float]
    utilitarian: float
    rawlsian: float
    nash: float
    atkinson: float  # at epsilon=0.5


def analyze_welfare(economy: ExchangeEconomy, allocation: Allocation) -> WelfareAnalysis:
    """Compute various welfare measures for an allocation."""
    u_a = economy.consumer_a.utility.utility_at(
        allocation.consumer_a['X'], allocation.consumer_a['Y'])
    u_b = economy.consumer_b.utility.utility_at(
        allocation.consumer_b['X'], allocation.consumer_b['Y'])
    
    utilities = [u_a, u_b]
    
    return WelfareAnalysis(
        utilities={economy.consumer_a.name: u_a, economy.consumer_b.name: u_b},
        utilitarian=utilitarian_welfare(utilities),
        rawlsian=rawlsian_welfare(utilities),
        nash=nash_welfare(utilities),
        atkinson=atkinson_welfare(utilities, epsilon=0.5)
    )


# =============================================================================
# INEQUALITY MEASURES
# =============================================================================

def gini_coefficient(values: List[float]) -> float:
    """
    Gini coefficient: measure of inequality.
    
    0 = perfect equality
    1 = perfect inequality
    """
    n = len(values)
    if n == 0:
        return 0
    
    sorted_vals = sorted(values)
    cumsum = np.cumsum(sorted_vals)
    
    # Gini = 1 - 2 * (area under Lorenz curve)
    # = (2 * Σ i*xi) / (n * Σ xi) - (n+1)/n
    numerator = sum((i + 1) * x for i, x in enumerate(sorted_vals))
    denominator = n * sum(sorted_vals)
    
    if denominator == 0:
        return 0
    
    return (2 * numerator / denominator) - (n + 1) / n


def coefficient_of_variation(values: List[float]) -> float:
    """CV = std / mean. Higher = more inequality."""
    if len(values) == 0 or np.mean(values) == 0:
        return 0
    return np.std(values) / np.mean(values)


def theil_index(values: List[float]) -> float:
    """
    Theil index: entropy-based inequality measure.
    
    T = (1/n) Σ (xi/μ) log(xi/μ)
    """
    n = len(values)
    if n == 0:
        return 0
    
    mu = np.mean(values)
    if mu == 0:
        return 0
    
    return sum((x / mu) * np.log(x / mu) for x in values if x > 0) / n


# =============================================================================
# WELFARE THEOREMS
# =============================================================================

@dataclass
class WelfareTheoremCheck:
    """Results of checking welfare theorems."""
    first_theorem_holds: bool
    second_theorem_applicable: bool
    explanation: str


def check_first_welfare_theorem(economy: ExchangeEconomy) -> WelfareTheoremCheck:
    """
    First Welfare Theorem: Competitive equilibrium → Pareto efficient.
    
    Conditions:
    1. Perfect competition (price-taking)
    2. Complete markets
    3. No externalities
    4. Local non-satiation (more is better)
    """
    eq = economy.find_equilibrium()
    
    if not eq.converged:
        return WelfareTheoremCheck(
            first_theorem_holds=False,
            second_theorem_applicable=False,
            explanation="Could not find equilibrium."
        )
    
    is_efficient = economy.is_pareto_efficient(eq.allocation)
    
    explanation = f"""
First Welfare Theorem Check
===========================
Equilibrium found: Yes
Equilibrium prices: Px={eq.prices['X']:.4f}, Py={eq.prices['Y']:.4f}

Pareto efficient: {is_efficient}

The First Welfare Theorem states that under standard assumptions
(perfect competition, complete markets, no externalities, local non-satiation),
any competitive equilibrium is Pareto efficient.

In this economy, the equilibrium allocation {'IS' if is_efficient else 'is NOT'} 
Pareto efficient, which {'confirms' if is_efficient else 'contradicts'} the theorem.
"""
    
    return WelfareTheoremCheck(
        first_theorem_holds=is_efficient,
        second_theorem_applicable=True,
        explanation=explanation
    )


def check_second_welfare_theorem(economy: ExchangeEconomy) -> WelfareTheoremCheck:
    """
    Second Welfare Theorem: Any Pareto efficient allocation can be achieved
    as a competitive equilibrium with appropriate lump-sum transfers.
    
    Conditions (in addition to First Theorem):
    1. Convex preferences
    2. Convex production sets
    """
    # Check that contract curve exists and contains multiple points
    contract = economy.contract_curve(n_points=20)
    
    if len(contract) < 2:
        return WelfareTheoremCheck(
            first_theorem_holds=True,
            second_theorem_applicable=False,
            explanation="Could not compute contract curve."
        )
    
    explanation = f"""
Second Welfare Theorem Check
============================
Contract curve computed: {len(contract)} Pareto efficient points found.

The Second Welfare Theorem states that any Pareto efficient allocation
can be achieved as a competitive equilibrium, provided we can make
appropriate lump-sum transfers of endowments.

This means:
1. Pick any point on the contract curve (any efficient allocation)
2. Redistribute endowments appropriately
3. Let markets work → achieve that allocation as equilibrium

The theorem requires convex preferences. With the utility functions
in this economy ({economy.consumer_a.utility.form_name}, {economy.consumer_b.utility.form_name}),
the theorem {'should apply' if 'cobb-douglas' in economy.consumer_a.utility.form_name.lower() else 'may or may not apply'}.
"""
    
    return WelfareTheoremCheck(
        first_theorem_holds=True,
        second_theorem_applicable=True,
        explanation=explanation
    )


# =============================================================================
# EFFICIENCY-EQUITY TRADEOFF
# =============================================================================

def utility_possibility_frontier(
    economy: ExchangeEconomy,
    n_points: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Utility Possibility Frontier.
    
    The UPF shows maximum utility for B given each level of utility for A.
    Points on the UPF are Pareto efficient.
    
    Returns (u_a_array, u_b_array).
    """
    contract = economy.contract_curve(n_points)
    
    u_a = []
    u_b = []
    
    for alloc in contract:
        ua = economy.consumer_a.utility.utility_at(
            alloc.consumer_a['X'], alloc.consumer_a['Y'])
        ub = economy.consumer_b.utility.utility_at(
            alloc.consumer_b['X'], alloc.consumer_b['Y'])
        u_a.append(ua)
        u_b.append(ub)
    
    return np.array(u_a), np.array(u_b)


def find_welfare_maximizing_allocation(
    economy: ExchangeEconomy,
    welfare_func: Callable[[List[float]], float] = utilitarian_welfare,
    n_points: int = 50
) -> Tuple[Allocation, float]:
    """
    Find the Pareto efficient allocation that maximizes given welfare function.
    
    Searches over the contract curve.
    """
    contract = economy.contract_curve(n_points)
    
    best_alloc = None
    best_welfare = -np.inf
    
    for alloc in contract:
        u_a = economy.consumer_a.utility.utility_at(
            alloc.consumer_a['X'], alloc.consumer_a['Y'])
        u_b = economy.consumer_b.utility.utility_at(
            alloc.consumer_b['X'], alloc.consumer_b['Y'])
        
        w = welfare_func([u_a, u_b])
        
        if w > best_welfare:
            best_welfare = w
            best_alloc = alloc
    
    return best_alloc, best_welfare
