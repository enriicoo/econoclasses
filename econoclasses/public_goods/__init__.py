"""
Public Goods Module for econoclasses

Provides analysis tools for:
- Public goods characteristics (non-rival, non-excludable)
- Free rider problem
- Lindahl equilibrium
- Voluntary contribution mechanisms
- Optimal public good provision (Samuelson condition)
- Club goods and congestion

Author: econoclasses
Version: 0.4.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict, Union
from scipy.optimize import minimize_scalar, brentq, minimize

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PublicGoodOutcome:
    """Results from public good analysis."""
    optimal_quantity: float
    private_quantity: float  # Under voluntary provision
    underprovision: float  # Optimal - Private
    total_benefit: float
    total_cost: float
    net_benefit: float
    individual_contributions: List[float]
    free_rider_loss: float
    
    def __repr__(self):
        return (f"PublicGoodOutcome\n"
                f"  Optimal Q={self.optimal_quantity:.2f}\n"
                f"  Private Q={self.private_quantity:.2f}\n"
                f"  Underprovision={self.underprovision:.2f}\n"
                f"  Free rider loss=${self.free_rider_loss:.2f}")


@dataclass
class LindahlResult:
    """Results from Lindahl equilibrium."""
    quantity: float
    personalized_prices: List[float]
    total_cost: float
    individual_payments: List[float]
    individual_surpluses: List[float]
    is_efficient: bool
    
    def __repr__(self):
        return (f"LindahlEquilibrium\n"
                f"  Q*={self.quantity:.2f}\n"
                f"  Personalized prices: {[f'${p:.2f}' for p in self.personalized_prices]}\n"
                f"  Payments: {[f'${p:.2f}' for p in self.individual_payments]}\n"
                f"  Efficient: {self.is_efficient}")


@dataclass
class ContributionGameResult:
    """Results from voluntary contribution game."""
    nash_contributions: List[float]
    total_provision: float
    individual_payoffs: List[float]
    efficient_provision: float
    efficiency_ratio: float  # Nash/Efficient
    is_free_rider_outcome: bool
    
    def __repr__(self):
        return (f"ContributionGame\n"
                f"  Nash contributions: {[f'{c:.2f}' for c in self.nash_contributions]}\n"
                f"  Total provision={self.total_provision:.2f}\n"
                f"  Efficient provision={self.efficient_provision:.2f}\n"
                f"  Efficiency={self.efficiency_ratio*100:.1f}%")


@dataclass
class ClubGoodResult:
    """Results from club good analysis."""
    optimal_membership: int
    optimal_quantity: float
    membership_fee: float
    per_member_benefit: float
    congestion_cost: float
    net_benefit_per_member: float
    
    def __repr__(self):
        return (f"ClubGood\n"
                f"  Optimal members={self.optimal_membership}\n"
                f"  Quantity={self.optimal_quantity:.2f}\n"
                f"  Fee=${self.membership_fee:.2f}/member\n"
                f"  Net benefit=${self.net_benefit_per_member:.2f}/member")


# ============================================================================
# PUBLIC GOOD CHARACTERISTICS
# ============================================================================

class PublicGood:
    """
    Represents a public good with consumer valuations.
    
    Public goods are:
    - Non-rival: One person's consumption doesn't reduce others'
    - Non-excludable: Cannot prevent anyone from consuming
    
    Parameters:
        marginal_benefits: List of individual MB functions MB_i(G)
        marginal_cost: MC function for providing the good
        num_consumers: Number of consumers (if MB functions not provided)
    """
    
    def __init__(
        self,
        marginal_benefits: List[Callable[[float], float]] = None,
        marginal_cost: Callable[[float], float] = None,
        num_consumers: int = 2
    ):
        if marginal_benefits is None:
            # Default: linear MB_i = a_i - b_i*G
            self.marginal_benefits = [
                lambda G, i=i: max(0, 10 - i - 0.5*G) 
                for i in range(num_consumers)
            ]
        else:
            self.marginal_benefits = marginal_benefits
            
        if marginal_cost is None:
            # Default: constant MC
            self.marginal_cost = lambda G: 5.0
        else:
            self.marginal_cost = marginal_cost
            
        self.n = len(self.marginal_benefits)
    
    def social_marginal_benefit(self, G: float) -> float:
        """
        Sum of individual marginal benefits (vertical summation).
        
        For public goods, we sum MB vertically (not horizontally like private goods)
        because everyone consumes the same quantity.
        """
        return sum(mb(G) for mb in self.marginal_benefits)
    
    def individual_mb(self, i: int, G: float) -> float:
        """Marginal benefit for individual i at quantity G."""
        return self.marginal_benefits[i](G)
    
    def optimal_quantity(self) -> float:
        """
        Find socially optimal quantity (Samuelson condition).
        
        Optimal where: Σ MB_i(G*) = MC(G*)
        """
        def objective(G):
            return (self.social_marginal_benefit(G) - self.marginal_cost(G))**2
        
        result = minimize_scalar(objective, bounds=(0, 100), method='bounded')
        return result.x
    
    def total_benefit(self, G: float) -> float:
        """Total benefit from quantity G (integral of social MB)."""
        from scipy.integrate import quad
        total = 0
        for mb in self.marginal_benefits:
            benefit, _ = quad(mb, 0, G)
            total += benefit
        return total
    
    def total_cost(self, G: float) -> float:
        """Total cost of providing quantity G."""
        from scipy.integrate import quad
        cost, _ = quad(self.marginal_cost, 0, G)
        return cost
    
    def net_benefit(self, G: float) -> float:
        """Net social benefit at quantity G."""
        return self.total_benefit(G) - self.total_cost(G)


# ============================================================================
# FREE RIDER PROBLEM
# ============================================================================

def voluntary_provision(
    public_good: PublicGood,
    incomes: List[float] = None
) -> PublicGoodOutcome:
    """
    Analyze voluntary (private) provision of a public good.
    
    Under voluntary provision:
    - Each individual contributes where their MB = MC
    - Results in underprovision (free rider problem)
    
    The Nash equilibrium has only the highest-value consumer contributing,
    and even they contribute too little.
    
    Returns:
        PublicGoodOutcome with voluntary vs optimal provision
    """
    n = public_good.n
    
    if incomes is None:
        incomes = [100.0] * n
    
    # Optimal quantity
    G_opt = public_good.optimal_quantity()
    
    # Under voluntary provision, each person decides independently
    # At equilibrium, person with highest MB contributes until their MB = MC
    # Others free ride (contribute 0)
    
    # Find highest MB at Q=0
    mb_at_zero = [public_good.individual_mb(i, 0) for i in range(n)]
    
    # The person with highest valuation contributes
    max_idx = np.argmax(mb_at_zero)
    
    # They contribute until MB = MC
    def equilibrium_condition(G):
        return public_good.individual_mb(max_idx, G) - public_good.marginal_cost(G)
    
    try:
        # Find where contributor's MB = MC
        G_private = brentq(equilibrium_condition, 0, 100)
    except:
        G_private = 0
    
    # All others contribute zero
    contributions = [0.0] * n
    contributions[max_idx] = G_private
    
    # Calculate welfare
    total_benefit_opt = public_good.total_benefit(G_opt)
    total_cost_opt = public_good.total_cost(G_opt)
    
    total_benefit_private = public_good.total_benefit(G_private)
    total_cost_private = public_good.total_cost(G_private)
    
    welfare_opt = total_benefit_opt - total_cost_opt
    welfare_private = total_benefit_private - total_cost_private
    
    free_rider_loss = welfare_opt - welfare_private
    
    return PublicGoodOutcome(
        optimal_quantity=G_opt,
        private_quantity=G_private,
        underprovision=G_opt - G_private,
        total_benefit=total_benefit_private,
        total_cost=total_cost_private,
        net_benefit=welfare_private,
        individual_contributions=contributions,
        free_rider_loss=free_rider_loss
    )


def contribution_game(
    n_players: int,
    endowment: float,
    mpcr: float,  # Marginal per-capita return
    contribution_max: float = None
) -> ContributionGameResult:
    """
    Analyze a public goods contribution game.
    
    Classic experimental setup:
    - n players each have endowment w
    - Each contributes c_i to public good
    - Public good return = mpcr * sum(c_i) to each player
    
    Payoff: π_i = (w - c_i) + mpcr * Σc_j
    
    Parameters:
        n_players: Number of players
        endowment: Initial endowment per player
        mpcr: Marginal per-capita return (typically 0.3-0.75)
        contribution_max: Maximum contribution (default: endowment)
        
    Returns:
        ContributionGameResult with Nash equilibrium
    """
    if contribution_max is None:
        contribution_max = endowment
    
    # Nash equilibrium: contribute 0 if mpcr < 1
    # Because dπ_i/dc_i = -1 + mpcr < 0
    if mpcr < 1:
        nash_contributions = [0.0] * n_players
        total_nash = 0
        is_free_rider = True
    else:
        # Full contribution is dominant
        nash_contributions = [contribution_max] * n_players
        total_nash = n_players * contribution_max
        is_free_rider = False
    
    # Nash payoffs
    nash_payoffs = [
        endowment - nash_contributions[i] + mpcr * total_nash
        for i in range(n_players)
    ]
    
    # Efficient provision: contribute fully if mpcr * n > 1
    if mpcr * n_players > 1:
        efficient_provision = n_players * contribution_max
        efficient_payoffs = [
            endowment - contribution_max + mpcr * efficient_provision
            for _ in range(n_players)
        ]
    else:
        efficient_provision = 0
        efficient_payoffs = [endowment] * n_players
    
    efficiency_ratio = total_nash / efficient_provision if efficient_provision > 0 else 0
    
    return ContributionGameResult(
        nash_contributions=nash_contributions,
        total_provision=total_nash,
        individual_payoffs=nash_payoffs,
        efficient_provision=efficient_provision,
        efficiency_ratio=efficiency_ratio,
        is_free_rider_outcome=is_free_rider
    )


# ============================================================================
# LINDAHL EQUILIBRIUM
# ============================================================================

def lindahl_equilibrium(public_good: PublicGood) -> LindahlResult:
    """
    Find Lindahl equilibrium for a public good.
    
    In Lindahl equilibrium:
    - Each consumer faces a personalized price equal to their MB
    - Sum of personalized prices equals MC
    - Resulting allocation is Pareto efficient
    
    Returns:
        LindahlResult with personalized prices and quantities
    """
    # Find optimal quantity first
    G_star = public_good.optimal_quantity()
    
    # Personalized prices = MB at optimal quantity
    personalized_prices = [
        public_good.individual_mb(i, G_star) 
        for i in range(public_good.n)
    ]
    
    # Verify Samuelson condition
    sum_prices = sum(personalized_prices)
    mc_at_optimal = public_good.marginal_cost(G_star)
    is_efficient = abs(sum_prices - mc_at_optimal) < 0.01
    
    # Individual payments = price * quantity
    individual_payments = [p * G_star for p in personalized_prices]
    
    # Total cost
    total_cost = public_good.total_cost(G_star)
    
    # Individual surpluses (integral of MB minus payment)
    individual_surpluses = []
    for i in range(public_good.n):
        from scipy.integrate import quad
        benefit, _ = quad(public_good.marginal_benefits[i], 0, G_star)
        surplus = benefit - individual_payments[i]
        individual_surpluses.append(surplus)
    
    return LindahlResult(
        quantity=G_star,
        personalized_prices=personalized_prices,
        total_cost=total_cost,
        individual_payments=individual_payments,
        individual_surpluses=individual_surpluses,
        is_efficient=is_efficient
    )


def compare_provision_mechanisms(public_good: PublicGood) -> Dict:
    """
    Compare different public good provision mechanisms.
    
    Mechanisms:
    1. Private (voluntary) provision - underprovision
    2. Lindahl equilibrium - efficient but impractical
    3. Majority voting - may not be efficient
    
    Returns:
        Dict comparing mechanisms
    """
    # Voluntary provision
    voluntary = voluntary_provision(public_good)
    
    # Lindahl
    lindahl = lindahl_equilibrium(public_good)
    
    # Majority voting (median voter)
    # Median voter's preferred quantity where their MB = their tax share * MC
    n = public_good.n
    
    # Under equal cost sharing, each pays MC/n
    def median_voter_condition(G):
        # Find median MB
        mbs = [public_good.individual_mb(i, G) for i in range(n)]
        median_mb = sorted(mbs)[n // 2]
        tax_share = public_good.marginal_cost(G) / n
        return median_mb - tax_share
    
    try:
        G_voting = brentq(median_voter_condition, 0, 100)
    except:
        G_voting = voluntary.private_quantity
    
    # Calculate welfare under voting
    welfare_voting = public_good.net_benefit(G_voting)
    
    return {
        'voluntary': {
            'quantity': voluntary.private_quantity,
            'welfare': voluntary.net_benefit,
            'description': 'Free rider problem leads to underprovision'
        },
        'lindahl': {
            'quantity': lindahl.quantity,
            'welfare': public_good.net_benefit(lindahl.quantity),
            'prices': lindahl.personalized_prices,
            'description': 'Efficient but requires knowing individual valuations'
        },
        'majority_voting': {
            'quantity': G_voting,
            'welfare': welfare_voting,
            'description': 'Median voter outcome under equal tax shares'
        },
        'optimal_quantity': voluntary.optimal_quantity,
        'optimal_welfare': public_good.net_benefit(voluntary.optimal_quantity)
    }


# ============================================================================
# CLUB GOODS
# ============================================================================

def club_good_equilibrium(
    benefit_per_member: Callable[[int, float], float],
    cost_function: Callable[[float], float],
    congestion_cost: Callable[[int], float],
    max_members: int = 100
) -> ClubGoodResult:
    """
    Find optimal club good provision and membership.
    
    Club goods are:
    - Excludable: Can charge membership fee
    - Partially rival: Congestion increases with members
    
    Parameters:
        benefit_per_member: Benefit function B(n, G) per member
        cost_function: Total cost C(G) of provision
        congestion_cost: Congestion cost per member as function of n
        max_members: Maximum possible membership
        
    Returns:
        ClubGoodResult with optimal club size and provision
    """
    best_welfare = float('-inf')
    best_n = 0
    best_G = 0
    best_fee = 0
    
    for n in range(1, max_members + 1):
        # For given n, find optimal G
        def net_benefit(G):
            if G <= 0:
                return 0
            total_benefit = n * benefit_per_member(n, G)
            total_cost = cost_function(G) + n * congestion_cost(n)
            return total_benefit - total_cost
        
        def neg_benefit(G):
            return -net_benefit(G)
        
        result = minimize_scalar(neg_benefit, bounds=(0, 100), method='bounded')
        G_opt = result.x
        welfare = net_benefit(G_opt)
        
        if welfare > best_welfare:
            best_welfare = welfare
            best_n = n
            best_G = G_opt
    
    # Calculate other metrics
    benefit = benefit_per_member(best_n, best_G)
    congestion = congestion_cost(best_n)
    total_cost = cost_function(best_G) + best_n * congestion
    
    # Membership fee = cost per member
    fee = total_cost / best_n
    net_per_member = benefit - fee
    
    return ClubGoodResult(
        optimal_membership=best_n,
        optimal_quantity=best_G,
        membership_fee=fee,
        per_member_benefit=benefit,
        congestion_cost=congestion,
        net_benefit_per_member=net_per_member
    )


def tiebout_competition(
    consumer_preferences: List[Callable[[float, float], float]],
    club_costs: List[Callable[[float], float]],
    num_clubs: int = 3
) -> Dict:
    """
    Tiebout model of local public good provision.
    
    Consumers "vote with their feet" by choosing which community to join
    based on public good levels and tax rates.
    
    Parameters:
        consumer_preferences: List of utility functions U(G, w-tax)
        club_costs: Cost functions for each club
        num_clubs: Number of competing clubs
        
    Returns:
        Dict with sorting equilibrium
    """
    n_consumers = len(consumer_preferences)
    
    # For simplicity, assume clubs offer different G levels
    # and consumers sort based on preferences
    
    # Create clubs with different public good levels
    G_levels = [5 + 5*i for i in range(num_clubs)]  # 5, 10, 15, ...
    
    # Each consumer chooses club that maximizes utility
    assignments = []
    endowment = 100  # Assume same endowment
    
    for i, pref in enumerate(consumer_preferences):
        best_club = 0
        best_utility = float('-inf')
        
        for j, G in enumerate(G_levels):
            # Tax = cost / number of members (estimated)
            # For now, assume equal sharing
            tax = club_costs[j % len(club_costs)](G) / (n_consumers / num_clubs)
            utility = pref(G, endowment - tax)
            
            if utility > best_utility:
                best_utility = utility
                best_club = j
        
        assignments.append(best_club)
    
    # Count members per club
    club_sizes = [assignments.count(j) for j in range(num_clubs)]
    
    # Recalculate taxes with actual membership
    club_taxes = []
    for j in range(num_clubs):
        if club_sizes[j] > 0:
            tax = club_costs[j % len(club_costs)](G_levels[j]) / club_sizes[j]
        else:
            tax = float('inf')
        club_taxes.append(tax)
    
    return {
        'assignments': assignments,
        'club_sizes': club_sizes,
        'G_levels': G_levels,
        'taxes': club_taxes,
        'description': 'Tiebout sorting: consumers choose communities based on G and taxes'
    }


# ============================================================================
# COMMON EXAMPLES
# ============================================================================

def national_defense_example() -> Dict:
    """
    Classic public good example: national defense.
    
    Pure public good: completely non-rival and non-excludable.
    """
    # Two types of citizens with different valuations
    def mb_high(G):
        return max(0, 100 - 2*G)  # High valuation
    
    def mb_low(G):
        return max(0, 50 - G)  # Low valuation
    
    def mc(G):
        return 10 + 0.5*G  # Increasing MC
    
    pg = PublicGood(
        marginal_benefits=[mb_high, mb_low],
        marginal_cost=mc
    )
    
    voluntary = voluntary_provision(pg)
    lindahl = lindahl_equilibrium(pg)
    
    return {
        'public_good': pg,
        'voluntary': voluntary,
        'lindahl': lindahl,
        'summary': (
            f"National Defense Example:\n"
            f"  Optimal provision: G*={voluntary.optimal_quantity:.1f}\n"
            f"  Voluntary provision: G={voluntary.private_quantity:.1f}\n"
            f"  Free rider loss: ${voluntary.free_rider_loss:.2f}\n"
            f"  Lindahl prices: High=${lindahl.personalized_prices[0]:.2f}, "
            f"Low=${lindahl.personalized_prices[1]:.2f}"
        )
    }


def local_park_example(num_residents: int = 100) -> Dict:
    """
    Local public good example: neighborhood park.
    
    Shows how voluntary provision fails with many consumers.
    """
    # Each resident has declining MB
    def mb_generator(i):
        # Heterogeneous valuations
        base = 10 - (i % 10) * 0.5
        return lambda G, b=base: max(0, b - 0.1*G)
    
    mbs = [mb_generator(i) for i in range(num_residents)]
    
    def mc(G):
        return 50 + 2*G  # High fixed + marginal cost
    
    pg = PublicGood(marginal_benefits=mbs, marginal_cost=mc)
    
    voluntary = voluntary_provision(pg)
    
    return {
        'public_good': pg,
        'voluntary': voluntary,
        'num_residents': num_residents,
        'summary': (
            f"Local Park ({num_residents} residents):\n"
            f"  Optimal: G*={voluntary.optimal_quantity:.1f}\n"
            f"  Voluntary: G={voluntary.private_quantity:.1f}\n"
            f"  Only 1 person contributes (highest valuation)\n"
            f"  {num_residents - 1} free riders\n"
            f"  Welfare loss: ${voluntary.free_rider_loss:.2f}"
        )
    }


def public_goods_game_experiment(
    n_players: int = 4,
    endowment: float = 20,
    mpcr: float = 0.4
) -> Dict:
    """
    Standard public goods game used in experiments.
    
    Parameters:
        n_players: Number of players (typically 4)
        endowment: Tokens per player (typically 20)
        mpcr: Marginal per-capita return (typically 0.4)
        
    Returns analysis of Nash prediction vs typical experimental results.
    """
    game = contribution_game(n_players, endowment, mpcr)
    
    # Typical experimental findings
    experimental_contribution_rate = 0.4  # ~40% initially
    experimental_contributions = [
        endowment * experimental_contribution_rate 
        for _ in range(n_players)
    ]
    exp_total = sum(experimental_contributions)
    
    # Calculate payoffs
    exp_payoffs = [
        endowment - experimental_contributions[i] + mpcr * exp_total
        for i in range(n_players)
    ]
    
    return {
        'game': game,
        'nash_prediction': {
            'contributions': game.nash_contributions,
            'total': game.total_provision,
            'payoffs': game.individual_payoffs
        },
        'typical_experiment': {
            'contributions': experimental_contributions,
            'total': exp_total,
            'payoffs': exp_payoffs
        },
        'efficiency': {
            'nash': game.efficiency_ratio,
            'experiment': exp_total / game.efficient_provision
        },
        'summary': (
            f"Public Goods Game (n={n_players}, MPCR={mpcr}):\n"
            f"  Nash prediction: contribute 0 (all free ride)\n"
            f"  Efficient: contribute {endowment} each\n"
            f"  Typical experiment: ~40% contribution rate\n"
            f"  Behavioral factors: altruism, reciprocity, confusion"
        )
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data classes
    'PublicGoodOutcome',
    'LindahlResult',
    'ContributionGameResult',
    'ClubGoodResult',
    
    # Main classes
    'PublicGood',
    
    # Free rider analysis
    'voluntary_provision',
    'contribution_game',
    
    # Lindahl equilibrium
    'lindahl_equilibrium',
    'compare_provision_mechanisms',
    
    # Club goods
    'club_good_equilibrium',
    'tiebout_competition',
    
    # Examples
    'national_defense_example',
    'local_park_example',
    'public_goods_game_experiment',
]
