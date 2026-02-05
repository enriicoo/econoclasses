"""
Market Structures Module for econoclasses

Provides analysis tools for different market structures:
- Perfect Competition
- Monopoly (with price discrimination)
- Monopolistic Competition
- Market Power Analysis
- Welfare Comparisons

Author: econoclasses
Version: 0.4.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad

# ============================================================================
# DATA CLASSES FOR RESULTS
# ============================================================================

@dataclass
class MarketOutcome:
    """Results from market equilibrium analysis."""
    price: float
    quantity: float
    firm_profit: float
    consumer_surplus: float
    producer_surplus: float
    total_surplus: float
    deadweight_loss: float = 0.0
    num_firms: int = 1
    market_structure: str = "unknown"
    
    def __repr__(self):
        return (f"MarketOutcome({self.market_structure})\n"
                f"  P={self.price:.2f}, Q={self.quantity:.2f}\n"
                f"  Profit={self.firm_profit:.2f}, CS={self.consumer_surplus:.2f}\n"
                f"  PS={self.producer_surplus:.2f}, TS={self.total_surplus:.2f}\n"
                f"  DWL={self.deadweight_loss:.2f}")


@dataclass 
class PriceDiscriminationResult:
    """Results from price discrimination analysis."""
    degree: int  # 1st, 2nd, or 3rd degree
    prices: List[float]
    quantities: List[float]
    total_quantity: float
    profit: float
    consumer_surplus: float
    total_surplus: float
    description: str
    
    def __repr__(self):
        return (f"PriceDiscrimination(degree={self.degree})\n"
                f"  Prices: {[f'{p:.2f}' for p in self.prices]}\n"
                f"  Quantities: {[f'{q:.2f}' for q in self.quantities]}\n"
                f"  Total Q={self.total_quantity:.2f}, Profit={self.profit:.2f}\n"
                f"  CS={self.consumer_surplus:.2f}, TS={self.total_surplus:.2f}")


@dataclass
class MonopolisticCompetitionResult:
    """Results from monopolistic competition equilibrium."""
    price: float
    quantity: float
    num_firms: int
    firm_profit: float
    markup: float
    average_cost: float
    excess_capacity: float  # efficient scale - actual output
    consumer_surplus: float
    
    def __repr__(self):
        return (f"MonopolisticCompetition\n"
                f"  P={self.price:.2f}, Q/firm={self.quantity:.2f}\n"
                f"  N={self.num_firms} firms, π={self.firm_profit:.2f}\n"
                f"  Markup={(self.markup-1)*100:.1f}%, AC={self.average_cost:.2f}\n"
                f"  Excess capacity={self.excess_capacity:.2f}")


# ============================================================================
# DEMAND AND COST CLASSES
# ============================================================================

class LinearDemand:
    """
    Linear demand curve: P = a - b*Q or Q = (a - P)/b
    
    Parameters:
        a: Intercept (choke price)
        b: Slope parameter (positive)
    """
    
    def __init__(self, a: float, b: float):
        if a <= 0:
            raise ValueError("Intercept 'a' must be positive")
        if b <= 0:
            raise ValueError("Slope 'b' must be positive")
        self.a = a
        self.b = b
        
    def price(self, Q: float) -> float:
        """Inverse demand: P(Q)"""
        return max(0, self.a - self.b * Q)
    
    def quantity(self, P: float) -> float:
        """Demand function: Q(P)"""
        if P >= self.a:
            return 0.0
        return (self.a - P) / self.b
    
    def marginal_revenue(self, Q: float) -> float:
        """MR = a - 2bQ for linear demand"""
        return self.a - 2 * self.b * Q
    
    def elasticity(self, P: float) -> float:
        """Price elasticity of demand at price P"""
        Q = self.quantity(P)
        if Q == 0:
            return float('-inf')
        return -(P / Q) * (1 / self.b)
    
    def consumer_surplus(self, P: float) -> float:
        """CS = 0.5 * (a - P) * Q"""
        Q = self.quantity(P)
        return 0.5 * (self.a - P) * Q
    
    def choke_price(self) -> float:
        """Price at which demand falls to zero"""
        return self.a
    
    def max_quantity(self) -> float:
        """Maximum quantity (at P=0)"""
        return self.a / self.b


class IsoelasticDemand:
    """
    Constant elasticity demand: Q = A * P^(-ε)
    
    Parameters:
        A: Scale parameter
        epsilon: Elasticity (absolute value, > 1 for valid monopoly)
    """
    
    def __init__(self, A: float, epsilon: float):
        if A <= 0:
            raise ValueError("Scale 'A' must be positive")
        if epsilon <= 0:
            raise ValueError("Elasticity must be positive")
        self.A = A
        self.epsilon = epsilon
        
    def quantity(self, P: float) -> float:
        """Q(P) = A * P^(-ε)"""
        if P <= 0:
            return float('inf')
        return self.A * P ** (-self.epsilon)
    
    def price(self, Q: float) -> float:
        """P(Q) = (Q/A)^(-1/ε)"""
        if Q <= 0:
            return float('inf')
        return (Q / self.A) ** (-1 / self.epsilon)
    
    def marginal_revenue(self, Q: float) -> float:
        """MR = P * (1 - 1/ε)"""
        P = self.price(Q)
        return P * (1 - 1 / self.epsilon)
    
    def elasticity(self, P: float = None) -> float:
        """Constant elasticity"""
        return -self.epsilon
    
    def consumer_surplus(self, P: float, P_max: float = 1000) -> float:
        """Approximate CS by integration"""
        result, _ = quad(lambda p: self.quantity(p), P, P_max)
        return result


class LinearCost:
    """
    Linear cost function: C(Q) = F + c*Q
    
    Parameters:
        c: Marginal cost (constant)
        F: Fixed cost
    """
    
    def __init__(self, c: float, F: float = 0):
        if c < 0:
            raise ValueError("Marginal cost must be non-negative")
        if F < 0:
            raise ValueError("Fixed cost must be non-negative")
        self.c = c
        self.F = F
        
    def total_cost(self, Q: float) -> float:
        """TC(Q) = F + c*Q"""
        if Q < 0:
            return self.F
        return self.F + self.c * Q
    
    def marginal_cost(self, Q: float = None) -> float:
        """MC = c (constant)"""
        return self.c
    
    def average_cost(self, Q: float) -> float:
        """AC(Q) = F/Q + c"""
        if Q <= 0:
            return float('inf')
        return self.F / Q + self.c
    
    def average_variable_cost(self, Q: float = None) -> float:
        """AVC = c"""
        return self.c


class QuadraticCost:
    """
    Quadratic cost function: C(Q) = F + c*Q + d*Q²
    
    Parameters:
        c: Linear cost coefficient
        d: Quadratic cost coefficient
        F: Fixed cost
    """
    
    def __init__(self, c: float, d: float, F: float = 0):
        if d < 0:
            raise ValueError("Quadratic coefficient must be non-negative")
        self.c = c
        self.d = d
        self.F = F
        
    def total_cost(self, Q: float) -> float:
        """TC(Q) = F + c*Q + d*Q²"""
        if Q < 0:
            return self.F
        return self.F + self.c * Q + self.d * Q**2
    
    def marginal_cost(self, Q: float) -> float:
        """MC(Q) = c + 2*d*Q"""
        return self.c + 2 * self.d * Q
    
    def average_cost(self, Q: float) -> float:
        """AC(Q) = F/Q + c + d*Q"""
        if Q <= 0:
            return float('inf')
        return self.F / Q + self.c + self.d * Q
    
    def efficient_scale(self) -> float:
        """Q where AC is minimized: Q* = sqrt(F/d)"""
        if self.d <= 0 or self.F <= 0:
            return 0.0
        return np.sqrt(self.F / self.d)


# ============================================================================
# PERFECT COMPETITION
# ============================================================================

def perfect_competition(
    demand: LinearDemand,
    cost: LinearCost,
    num_firms: int = 100
) -> MarketOutcome:
    """
    Solve for perfect competition equilibrium.
    
    In perfect competition:
    - P = MC (marginal cost pricing)
    - Firms are price takers
    - Free entry/exit drives economic profit to zero in long run
    
    Parameters:
        demand: Linear demand curve
        cost: Cost function (assumed same for all firms)
        num_firms: Number of firms (for profit calculation)
        
    Returns:
        MarketOutcome with equilibrium values
    """
    # P = MC
    P_star = cost.marginal_cost()
    Q_star = demand.quantity(P_star)
    
    # Check if market exists
    if Q_star <= 0:
        return MarketOutcome(
            price=P_star,
            quantity=0,
            firm_profit=0,
            consumer_surplus=0,
            producer_surplus=0,
            total_surplus=0,
            deadweight_loss=0,
            num_firms=num_firms,
            market_structure="perfect_competition"
        )
    
    # Per-firm quantity
    q_firm = Q_star / num_firms
    
    # Profits
    revenue = P_star * q_firm
    tc = cost.total_cost(q_firm)
    firm_profit = revenue - tc
    
    # Surplus
    CS = demand.consumer_surplus(P_star)
    # PS = integral from 0 to Q of (P - MC) = (P - c) * Q for linear cost
    PS = (P_star - cost.c) * Q_star
    TS = CS + PS
    
    return MarketOutcome(
        price=P_star,
        quantity=Q_star,
        firm_profit=firm_profit,
        consumer_surplus=CS,
        producer_surplus=PS,
        total_surplus=TS,
        deadweight_loss=0,
        num_firms=num_firms,
        market_structure="perfect_competition"
    )


# ============================================================================
# MONOPOLY
# ============================================================================

def monopoly(
    demand: LinearDemand,
    cost: LinearCost
) -> MarketOutcome:
    """
    Solve for monopoly equilibrium.
    
    Monopolist sets MR = MC:
    - For linear demand P = a - bQ: MR = a - 2bQ
    - Optimal Q: a - 2bQ = c → Q* = (a - c)/(2b)
    - Price: P* = a - bQ* = (a + c)/2
    
    Parameters:
        demand: Linear demand curve
        cost: Cost function
        
    Returns:
        MarketOutcome with equilibrium values
    """
    a, b = demand.a, demand.b
    c = cost.marginal_cost()
    
    # Check if monopoly is viable
    if c >= a:
        return MarketOutcome(
            price=a,
            quantity=0,
            firm_profit=-cost.F if hasattr(cost, 'F') else 0,
            consumer_surplus=0,
            producer_surplus=0,
            total_surplus=0,
            deadweight_loss=0,
            num_firms=1,
            market_structure="monopoly"
        )
    
    # Monopoly solution
    Q_m = (a - c) / (2 * b)
    P_m = a - b * Q_m  # = (a + c) / 2
    
    # Competitive benchmark for DWL
    Q_c = (a - c) / b
    P_c = c
    
    # Profits
    revenue = P_m * Q_m
    tc = cost.total_cost(Q_m)
    profit = revenue - tc
    
    # Surplus calculations
    CS = demand.consumer_surplus(P_m)
    PS = (P_m - c) * Q_m  # Monopoly producer surplus
    
    # Total surplus under monopoly
    TS_monopoly = CS + PS
    
    # Competitive total surplus (maximum possible)
    CS_c = demand.consumer_surplus(P_c)
    PS_c = 0  # Zero economic profit in competition with P = MC
    TS_competitive = CS_c + PS_c
    
    # Deadweight loss
    DWL = TS_competitive - TS_monopoly
    # Alternative formula: DWL = 0.5 * (P_m - c) * (Q_c - Q_m)
    
    return MarketOutcome(
        price=P_m,
        quantity=Q_m,
        firm_profit=profit,
        consumer_surplus=CS,
        producer_surplus=PS,
        total_surplus=TS_monopoly,
        deadweight_loss=DWL,
        num_firms=1,
        market_structure="monopoly"
    )


def monopoly_with_elasticity(
    demand: IsoelasticDemand,
    cost: LinearCost
) -> MarketOutcome:
    """
    Monopoly pricing with constant elasticity demand.
    
    Lerner Index: (P - MC)/P = 1/|ε|
    → P = MC / (1 - 1/|ε|) = MC * |ε| / (|ε| - 1)
    
    Parameters:
        demand: Isoelastic demand curve
        cost: Cost function
        
    Returns:
        MarketOutcome with equilibrium values
    """
    epsilon = demand.epsilon
    c = cost.marginal_cost()
    
    if epsilon <= 1:
        raise ValueError("Elasticity must be > 1 for monopoly to exist")
    
    # Monopoly markup
    markup = epsilon / (epsilon - 1)
    P_m = c * markup
    Q_m = demand.quantity(P_m)
    
    # Profit
    profit = (P_m - c) * Q_m - cost.F
    
    # Approximate CS (integrate from P_m to some high price)
    CS = demand.consumer_surplus(P_m)
    
    return MarketOutcome(
        price=P_m,
        quantity=Q_m,
        firm_profit=profit,
        consumer_surplus=CS,
        producer_surplus=(P_m - c) * Q_m,
        total_surplus=CS + (P_m - c) * Q_m,
        num_firms=1,
        market_structure="monopoly_isoelastic"
    )


# ============================================================================
# PRICE DISCRIMINATION
# ============================================================================

def first_degree_discrimination(
    demand: LinearDemand,
    cost: LinearCost
) -> PriceDiscriminationResult:
    """
    First-degree (perfect) price discrimination.
    
    Monopolist captures entire consumer surplus by charging
    each consumer their reservation price.
    
    Result:
    - Q = competitive quantity (where P = MC)
    - Profit = entire social surplus
    - CS = 0
    - No deadweight loss (efficient)
    """
    a, b = demand.a, demand.b
    c = cost.marginal_cost()
    
    # Output at competitive level (efficient)
    Q = (a - c) / b
    
    # Profit = integral from 0 to Q of (P(q) - c) dq
    # = integral of (a - bq - c) dq = (a-c)Q - 0.5*b*Q^2
    profit = (a - c) * Q - 0.5 * b * Q**2 - cost.F
    
    # All surplus goes to producer
    CS = 0
    TS = profit + cost.F  # Total surplus
    
    return PriceDiscriminationResult(
        degree=1,
        prices=[f"P(q) = {a} - {b}q (continuous)"],
        quantities=[Q],
        total_quantity=Q,
        profit=profit,
        consumer_surplus=CS,
        total_surplus=TS,
        description="Perfect price discrimination: Each unit sold at reservation price"
    )


def third_degree_discrimination(
    demands: List[LinearDemand],
    cost: LinearCost,
    market_names: List[str] = None
) -> PriceDiscriminationResult:
    """
    Third-degree price discrimination across separate markets.
    
    Monopolist charges different prices in different markets
    based on demand elasticities.
    
    Optimal: MR_i = MC for each market
    
    Parameters:
        demands: List of demand curves for different markets
        cost: Cost function (same MC for all markets)
        market_names: Optional names for markets
        
    Returns:
        PriceDiscriminationResult with prices and quantities per market
    """
    if market_names is None:
        market_names = [f"Market {i+1}" for i in range(len(demands))]
    
    c = cost.marginal_cost()
    prices = []
    quantities = []
    
    for demand in demands:
        a, b = demand.a, demand.b
        # MR = MC: a - 2bQ = c → Q = (a-c)/(2b)
        if c >= a:
            Q_i = 0
            P_i = a
        else:
            Q_i = (a - c) / (2 * b)
            P_i = a - b * Q_i
        prices.append(P_i)
        quantities.append(Q_i)
    
    total_Q = sum(quantities)
    
    # Profit
    revenue = sum(p * q for p, q in zip(prices, quantities))
    total_cost = cost.total_cost(total_Q)
    profit = revenue - total_cost
    
    # Consumer surplus in each market
    CS = sum(d.consumer_surplus(p) for d, p in zip(demands, prices))
    
    # Total surplus
    TS = CS + profit + cost.F
    
    desc_parts = [f"{name}: P={p:.2f}, Q={q:.2f}" 
                  for name, p, q in zip(market_names, prices, quantities)]
    description = "Third-degree discrimination:\n" + "\n".join(desc_parts)
    
    return PriceDiscriminationResult(
        degree=3,
        prices=prices,
        quantities=quantities,
        total_quantity=total_Q,
        profit=profit,
        consumer_surplus=CS,
        total_surplus=TS,
        description=description
    )


def two_part_tariff(
    demand: LinearDemand,
    cost: LinearCost,
    num_consumers: int = 1
) -> Dict:
    """
    Two-part tariff pricing: Entry fee (F) + per-unit price (p).
    
    Optimal strategy: Set p = MC, then extract all CS as entry fee.
    
    Parameters:
        demand: Individual consumer demand
        cost: Cost function
        num_consumers: Number of identical consumers
        
    Returns:
        Dict with pricing strategy and outcomes
    """
    a, b = demand.a, demand.b
    c = cost.marginal_cost()
    
    # Set per-unit price at marginal cost
    p_unit = c
    Q_per_consumer = demand.quantity(p_unit)
    
    # Entry fee = consumer surplus at p = MC
    entry_fee = demand.consumer_surplus(p_unit)
    
    # Total profit
    total_Q = num_consumers * Q_per_consumer
    profit = num_consumers * entry_fee - cost.F
    
    return {
        'unit_price': p_unit,
        'entry_fee': entry_fee,
        'quantity_per_consumer': Q_per_consumer,
        'total_quantity': total_Q,
        'profit': profit,
        'consumer_surplus': 0,  # All extracted
        'total_surplus': profit + cost.F,
        'description': f"Two-part tariff: Fee=${entry_fee:.2f}, Price=${p_unit:.2f}/unit"
    }


# ============================================================================
# MONOPOLISTIC COMPETITION
# ============================================================================

def monopolistic_competition(
    market_demand: LinearDemand,
    cost: QuadraticCost,
    product_differentiation: float = 0.5
) -> MonopolisticCompetitionResult:
    """
    Monopolistic competition equilibrium (Chamberlin model).
    
    Features:
    - Many firms with differentiated products
    - Each firm faces downward-sloping demand
    - Free entry drives profits to zero
    - Excess capacity in equilibrium
    
    Parameters:
        market_demand: Total market demand
        cost: Firm cost function (quadratic for U-shaped AC)
        product_differentiation: Parameter [0,1] affecting demand elasticity
        
    Returns:
        MonopolisticCompetitionResult with equilibrium values
    """
    # Firm's perceived demand is more elastic than market demand
    # d_i = a_i - b_i * q_i where elasticity depends on differentiation
    
    # For simplicity, assume symmetric firms
    # In equilibrium: P = AC (zero profit) and MR = MC (profit max)
    
    # Efficient scale
    q_efficient = cost.efficient_scale()
    ac_min = cost.average_cost(q_efficient)
    
    # Find equilibrium where P = AC and firm maximizes profit
    # With differentiation, firm produces below efficient scale
    
    # Demand facing each firm depends on number of firms
    # Iteratively solve for equilibrium number of firms
    
    a, b = market_demand.a, market_demand.b
    
    # Start with guess for number of firms
    def find_equilibrium(n_firms):
        if n_firms < 1:
            return float('inf'), 0, 0
        
        # Each firm's demand: q = (a/n) - b_firm * p
        # where b_firm depends on substitutability
        a_firm = a / n_firms
        b_firm = b * (1 + product_differentiation * (n_firms - 1)) / n_firms
        
        # Monopoly pricing on firm's demand curve
        # MR = MC: a_firm - 2*b_firm*q = MC(q)
        # a_firm - 2*b_firm*q = c + 2*d*q
        # q = (a_firm - c) / (2*b_firm + 2*d)
        
        c = cost.c
        d = cost.d
        
        q = (a_firm - c) / (2 * b_firm + 2 * d)
        if q <= 0:
            return float('inf'), 0, 0
            
        p = a_firm - b_firm * q
        ac = cost.average_cost(q)
        profit = (p - ac) * q
        
        return profit, q, p
    
    # Binary search for zero-profit equilibrium
    n_low, n_high = 1, 1000
    
    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        profit, _, _ = find_equilibrium(n_mid)
        if profit > 0.01:  # Positive profit attracts entry
            n_low = n_mid
        else:
            n_high = n_mid
    
    # Get equilibrium values
    _, q_star, p_star = find_equilibrium(n_high)
    
    # Recalculate with found n
    n_firms = n_high
    ac_star = cost.average_cost(q_star)
    mc_star = cost.marginal_cost(q_star)
    markup = p_star / mc_star if mc_star > 0 else float('inf')
    excess_capacity = q_efficient - q_star
    
    # Consumer surplus (approximate)
    total_Q = n_firms * q_star
    CS = market_demand.consumer_surplus(p_star)
    
    return MonopolisticCompetitionResult(
        price=p_star,
        quantity=q_star,
        num_firms=n_firms,
        firm_profit=0,  # Zero in equilibrium
        markup=markup,
        average_cost=ac_star,
        excess_capacity=max(0, excess_capacity),
        consumer_surplus=CS
    )


# ============================================================================
# MARKET POWER ANALYSIS
# ============================================================================

def lerner_index(price: float, marginal_cost: float) -> float:
    """
    Calculate the Lerner Index of market power.
    
    L = (P - MC) / P
    
    Interpretation:
    - L = 0: Perfect competition
    - L = 1: Maximum market power (MC = 0)
    - L = 1/|ε|: Relationship to demand elasticity
    """
    if price <= 0:
        return 0.0
    return (price - marginal_cost) / price


def herfindahl_index(market_shares: List[float]) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI).
    
    HHI = Σ(s_i)² where s_i is market share (as percentage)
    
    Interpretation:
    - HHI < 1500: Competitive market
    - 1500 < HHI < 2500: Moderately concentrated
    - HHI > 2500: Highly concentrated
    - HHI = 10000: Monopoly
    
    Parameters:
        market_shares: List of market shares (as decimals summing to 1)
    """
    # Convert to percentages
    shares_pct = [s * 100 for s in market_shares]
    return sum(s**2 for s in shares_pct)


def concentration_ratio(market_shares: List[float], n: int = 4) -> float:
    """
    Calculate n-firm concentration ratio.
    
    CR_n = sum of market shares of n largest firms
    
    Parameters:
        market_shares: List of market shares (as decimals)
        n: Number of firms to include (default 4)
    """
    sorted_shares = sorted(market_shares, reverse=True)
    return sum(sorted_shares[:n])


def market_power_analysis(
    prices: List[float],
    quantities: List[float],
    marginal_costs: List[float]
) -> Dict:
    """
    Comprehensive market power analysis for an industry.
    
    Parameters:
        prices: List of prices charged by each firm
        quantities: List of quantities sold by each firm
        marginal_costs: List of marginal costs for each firm
        
    Returns:
        Dict with various market power measures
    """
    total_Q = sum(quantities)
    market_shares = [q / total_Q for q in quantities]
    
    # Weighted average price
    avg_price = sum(p * s for p, s in zip(prices, market_shares))
    
    # Weighted average MC
    avg_mc = sum(mc * s for mc, s in zip(marginal_costs, market_shares))
    
    # Individual Lerner indices
    lerner_indices = [lerner_index(p, mc) for p, mc in zip(prices, marginal_costs)]
    
    # Industry Lerner index
    industry_lerner = lerner_index(avg_price, avg_mc)
    
    return {
        'hhi': herfindahl_index(market_shares),
        'cr4': concentration_ratio(market_shares, 4),
        'cr8': concentration_ratio(market_shares, 8),
        'lerner_indices': lerner_indices,
        'industry_lerner': industry_lerner,
        'market_shares': market_shares,
        'num_firms': len(prices),
        'interpretation': _interpret_concentration(herfindahl_index(market_shares))
    }


def _interpret_concentration(hhi: float) -> str:
    """Interpret HHI value."""
    if hhi < 1500:
        return "Competitive market (HHI < 1500)"
    elif hhi < 2500:
        return "Moderately concentrated (1500 < HHI < 2500)"
    else:
        return "Highly concentrated (HHI > 2500)"


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_market_structures(
    demand: LinearDemand,
    cost: LinearCost
) -> Dict[str, MarketOutcome]:
    """
    Compare outcomes across market structures.
    
    Returns dict with outcomes for:
    - Perfect competition
    - Monopoly
    - First-degree price discrimination
    """
    pc = perfect_competition(demand, cost)
    mon = monopoly(demand, cost)
    
    # First-degree discrimination (efficient)
    fpd = first_degree_discrimination(demand, cost)
    fpd_outcome = MarketOutcome(
        price=cost.marginal_cost(),  # Marginal price
        quantity=fpd.total_quantity,
        firm_profit=fpd.profit,
        consumer_surplus=fpd.consumer_surplus,
        producer_surplus=fpd.profit + cost.F,
        total_surplus=fpd.total_surplus,
        deadweight_loss=0,
        num_firms=1,
        market_structure="first_degree_discrimination"
    )
    
    return {
        'perfect_competition': pc,
        'monopoly': mon,
        'first_degree_discrimination': fpd_outcome
    }


def welfare_comparison_table(
    demand: LinearDemand,
    cost: LinearCost
) -> str:
    """
    Generate formatted comparison table of market structures.
    """
    results = compare_market_structures(demand, cost)
    
    lines = [
        "=" * 70,
        "MARKET STRUCTURE COMPARISON",
        "=" * 70,
        f"{'Metric':<25} {'Competition':<15} {'Monopoly':<15} {'1st Deg PD':<15}",
        "-" * 70
    ]
    
    pc = results['perfect_competition']
    mon = results['monopoly']
    fpd = results['first_degree_discrimination']
    
    metrics = [
        ('Price', pc.price, mon.price, f"{cost.marginal_cost():.2f}*"),
        ('Quantity', pc.quantity, mon.quantity, fpd.quantity),
        ('Consumer Surplus', pc.consumer_surplus, mon.consumer_surplus, fpd.consumer_surplus),
        ('Producer Surplus', pc.producer_surplus, mon.producer_surplus, fpd.producer_surplus),
        ('Total Surplus', pc.total_surplus, mon.total_surplus, fpd.total_surplus),
        ('Deadweight Loss', pc.deadweight_loss, mon.deadweight_loss, fpd.deadweight_loss),
        ('Firm Profit', pc.firm_profit, mon.firm_profit, fpd.firm_profit),
    ]
    
    for name, val_pc, val_mon, val_fpd in metrics:
        if isinstance(val_fpd, str):
            lines.append(f"{name:<25} {val_pc:<15.2f} {val_mon:<15.2f} {val_fpd:<15}")
        else:
            lines.append(f"{name:<25} {val_pc:<15.2f} {val_mon:<15.2f} {val_fpd:<15.2f}")
    
    lines.extend([
        "-" * 70,
        "* First-degree PD: continuous pricing from choke price down to MC",
        "=" * 70
    ])
    
    return "\n".join(lines)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data classes
    'MarketOutcome',
    'PriceDiscriminationResult', 
    'MonopolisticCompetitionResult',
    
    # Demand and cost
    'LinearDemand',
    'IsoelasticDemand',
    'LinearCost',
    'QuadraticCost',
    
    # Market structures
    'perfect_competition',
    'monopoly',
    'monopoly_with_elasticity',
    'monopolistic_competition',
    
    # Price discrimination
    'first_degree_discrimination',
    'third_degree_discrimination',
    'two_part_tariff',
    
    # Market power
    'lerner_index',
    'herfindahl_index',
    'concentration_ratio',
    'market_power_analysis',
    
    # Comparison
    'compare_market_structures',
    'welfare_comparison_table',
]
