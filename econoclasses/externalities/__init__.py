"""
Externalities Module for econoclasses

Provides analysis tools for:
- Negative externalities (pollution, congestion)
- Positive externalities (education, R&D spillovers)
- Pigouvian taxes and subsidies
- Coase theorem and bargaining solutions
- Cap-and-trade systems
- Welfare analysis with externalities

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
class ExternalityOutcome:
    """Results from externality analysis."""
    quantity_market: float  # Private market outcome
    quantity_optimal: float  # Socially optimal outcome
    price_market: float
    price_optimal: float
    external_cost: float  # Total external cost at market Q
    deadweight_loss: float
    welfare_market: float
    welfare_optimal: float
    externality_type: str  # 'negative' or 'positive'
    
    def __repr__(self):
        return (f"ExternalityOutcome({self.externality_type})\n"
                f"  Market: Q={self.quantity_market:.2f}, P={self.price_market:.2f}\n"
                f"  Optimal: Q={self.quantity_optimal:.2f}, P={self.price_optimal:.2f}\n"
                f"  External cost={self.external_cost:.2f}, DWL={self.deadweight_loss:.2f}")


@dataclass
class PigouvianTaxResult:
    """Results from Pigouvian tax/subsidy analysis."""
    tax_rate: float  # Optimal tax (negative = subsidy)
    quantity_after_tax: float
    price_consumers: float
    price_producers: float
    tax_revenue: float
    welfare_gain: float
    description: str
    
    def __repr__(self):
        policy = "tax" if self.tax_rate >= 0 else "subsidy"
        return (f"PigouvianTax({policy}=${abs(self.tax_rate):.2f}/unit)\n"
                f"  Q={self.quantity_after_tax:.2f}\n"
                f"  P_consumers={self.price_consumers:.2f}, P_producers={self.price_producers:.2f}\n"
                f"  Revenue={self.tax_revenue:.2f}, Welfare gain={self.welfare_gain:.2f}")


@dataclass
class CoaseResult:
    """Results from Coase theorem analysis."""
    efficient_quantity: float
    payment: float  # Side payment in bargaining
    polluter_payoff: float
    victim_payoff: float
    total_surplus: float
    property_rights_holder: str
    description: str
    
    def __repr__(self):
        return (f"CoaseResult(rights={self.property_rights_holder})\n"
                f"  Efficient Q={self.efficient_quantity:.2f}\n"
                f"  Payment={self.payment:.2f}\n"
                f"  Polluter payoff={self.polluter_payoff:.2f}\n"
                f"  Victim payoff={self.victim_payoff:.2f}")


@dataclass
class CapAndTradeResult:
    """Results from cap-and-trade analysis."""
    cap: float  # Total emissions cap
    permit_price: float
    firm_emissions: List[float]
    firm_abatement: List[float]
    firm_permit_trades: List[float]  # Positive = buy, negative = sell
    total_cost: float
    cost_savings_vs_uniform: float
    
    def __repr__(self):
        return (f"CapAndTrade(cap={self.cap:.1f})\n"
                f"  Permit price=${self.permit_price:.2f}\n"
                f"  Total abatement cost=${self.total_cost:.2f}\n"
                f"  Savings vs uniform=${self.cost_savings_vs_uniform:.2f}")


# ============================================================================
# EXTERNALITY FUNCTIONS
# ============================================================================

class ExternalityMarket:
    """
    Market with externalities.
    
    Parameters:
        demand_intercept: a in P = a - b*Q (demand)
        demand_slope: b in P = a - b*Q
        supply_intercept: c in P = c + d*Q (private MC)
        supply_slope: d in P = c + d*Q
        external_cost: e in MEC = e*Q (marginal external cost, or constant)
        external_cost_type: 'linear' (MEC = e*Q) or 'constant' (MEC = e)
        externality_type: 'negative' or 'positive'
    """
    
    def __init__(
        self,
        demand_intercept: float,
        demand_slope: float,
        supply_intercept: float,
        supply_slope: float,
        external_cost: float,
        external_cost_type: str = 'constant',
        externality_type: str = 'negative'
    ):
        self.a = demand_intercept
        self.b = demand_slope
        self.c = supply_intercept
        self.d = supply_slope
        self.e = external_cost
        self.ec_type = external_cost_type
        self.ext_type = externality_type
        
    def demand(self, Q: float) -> float:
        """Inverse demand: P = a - b*Q"""
        return max(0, self.a - self.b * Q)
    
    def private_mc(self, Q: float) -> float:
        """Private marginal cost: MC = c + d*Q"""
        return self.c + self.d * Q
    
    def marginal_external_cost(self, Q: float) -> float:
        """Marginal external cost (or benefit if positive externality)."""
        if self.ec_type == 'linear':
            return self.e * Q
        else:
            return self.e
    
    def social_mc(self, Q: float) -> float:
        """Social marginal cost = Private MC + MEC (or - MEB)"""
        mec = self.marginal_external_cost(Q)
        if self.ext_type == 'negative':
            return self.private_mc(Q) + mec
        else:
            return self.private_mc(Q) - mec
    
    def total_external_cost(self, Q: float) -> float:
        """Total external cost up to quantity Q."""
        if self.ec_type == 'linear':
            return 0.5 * self.e * Q**2
        else:
            return self.e * Q
    
    def market_equilibrium(self) -> Tuple[float, float]:
        """
        Private market equilibrium (ignoring externality).
        Demand = Private Supply: a - b*Q = c + d*Q
        """
        Q_market = (self.a - self.c) / (self.b + self.d)
        P_market = self.a - self.b * Q_market
        return max(0, Q_market), max(0, P_market)
    
    def social_optimum(self) -> Tuple[float, float]:
        """
        Socially optimal equilibrium.
        Demand = Social MC
        """
        if self.ext_type == 'negative':
            if self.ec_type == 'linear':
                # a - b*Q = c + d*Q + e*Q = c + (d+e)*Q
                Q_opt = (self.a - self.c) / (self.b + self.d + self.e)
            else:
                # a - b*Q = c + d*Q + e
                Q_opt = (self.a - self.c - self.e) / (self.b + self.d)
        else:  # positive externality
            if self.ec_type == 'linear':
                # a - b*Q = c + d*Q - e*Q = c + (d-e)*Q
                denom = self.b + self.d - self.e
                if denom <= 0:
                    Q_opt = self.a / self.b  # Max Q
                else:
                    Q_opt = (self.a - self.c) / denom
            else:
                # a - b*Q = c + d*Q - e → a + e = c + (b+d)*Q
                Q_opt = (self.a - self.c + self.e) / (self.b + self.d)
        
        P_opt = self.demand(Q_opt)
        return max(0, Q_opt), max(0, P_opt)
    
    def analyze(self) -> ExternalityOutcome:
        """Complete externality analysis."""
        Q_m, P_m = self.market_equilibrium()
        Q_opt, P_opt = self.social_optimum()
        
        # External cost at market quantity
        ext_cost = self.total_external_cost(Q_m)
        
        # Welfare calculations
        # Market welfare = CS + PS - External cost
        CS_market = 0.5 * (self.a - P_m) * Q_m
        PS_market = 0.5 * (P_m - self.c) * Q_m
        W_market = CS_market + PS_market
        if self.ext_type == 'negative':
            W_market -= ext_cost
        else:
            W_market += ext_cost
        
        # Optimal welfare
        CS_opt = 0.5 * (self.a - P_opt) * Q_opt
        PS_opt = 0.5 * (P_opt - self.c) * Q_opt
        ext_cost_opt = self.total_external_cost(Q_opt)
        W_opt = CS_opt + PS_opt
        if self.ext_type == 'negative':
            W_opt -= ext_cost_opt
        else:
            W_opt += ext_cost_opt
        
        # Deadweight loss
        DWL = W_opt - W_market
        
        return ExternalityOutcome(
            quantity_market=Q_m,
            quantity_optimal=Q_opt,
            price_market=P_m,
            price_optimal=P_opt,
            external_cost=ext_cost,
            deadweight_loss=abs(DWL),
            welfare_market=W_market,
            welfare_optimal=W_opt,
            externality_type=self.ext_type
        )


# ============================================================================
# PIGOUVIAN TAXES AND SUBSIDIES
# ============================================================================

def pigouvian_tax(market: ExternalityMarket) -> PigouvianTaxResult:
    """
    Calculate optimal Pigouvian tax/subsidy.
    
    For negative externality: Tax = MEC at optimal Q
    For positive externality: Subsidy = MEB at optimal Q
    
    Returns:
        PigouvianTaxResult with optimal policy
    """
    Q_opt, P_opt = market.social_optimum()
    Q_m, P_m = market.market_equilibrium()
    
    # Optimal tax = MEC at social optimum
    mec_opt = market.marginal_external_cost(Q_opt)
    
    if market.ext_type == 'negative':
        tax = mec_opt
        description = f"Pigouvian tax of ${tax:.2f}/unit corrects negative externality"
    else:
        tax = -mec_opt  # Negative tax = subsidy
        description = f"Pigouvian subsidy of ${abs(tax):.2f}/unit corrects positive externality"
    
    # After-tax prices
    P_consumers = P_opt  # Consumers pay demand price
    P_producers = market.private_mc(Q_opt)  # Producers receive supply price
    
    # Tax revenue
    revenue = tax * Q_opt
    
    # Welfare gain
    outcome = market.analyze()
    welfare_gain = outcome.welfare_optimal - outcome.welfare_market
    
    return PigouvianTaxResult(
        tax_rate=tax,
        quantity_after_tax=Q_opt,
        price_consumers=P_consumers,
        price_producers=P_producers,
        tax_revenue=revenue,
        welfare_gain=welfare_gain,
        description=description
    )


def tax_incidence_with_externality(
    market: ExternalityMarket,
    tax: float
) -> Dict:
    """
    Analyze incidence of a specific tax rate (not necessarily optimal).
    
    Parameters:
        market: ExternalityMarket
        tax: Per-unit tax
        
    Returns:
        Dict with incidence analysis
    """
    # New equilibrium: P_d = a - b*Q, P_s = c + d*Q, P_d = P_s + tax
    # a - b*Q = c + d*Q + tax
    Q_tax = (market.a - market.c - tax) / (market.b + market.d)
    Q_tax = max(0, Q_tax)
    
    P_demand = market.demand(Q_tax)
    P_supply = market.private_mc(Q_tax)
    
    # Compare to no-tax equilibrium
    Q_m, P_m = market.market_equilibrium()
    
    # Tax burden
    consumer_burden = P_demand - P_m
    producer_burden = P_m - P_supply
    
    # Elasticity-based incidence
    elasticity_demand = market.b  # Slope of demand
    elasticity_supply = market.d  # Slope of supply
    consumer_share = elasticity_supply / (elasticity_demand + elasticity_supply)
    producer_share = elasticity_demand / (elasticity_demand + elasticity_supply)
    
    return {
        'quantity': Q_tax,
        'price_consumers': P_demand,
        'price_producers': P_supply,
        'tax_revenue': tax * Q_tax,
        'consumer_burden': consumer_burden,
        'producer_burden': producer_burden,
        'consumer_share': consumer_share,
        'producer_share': producer_share,
        'deadweight_loss_from_tax': 0.5 * tax * (Q_m - Q_tax)
    }


# ============================================================================
# COASE THEOREM
# ============================================================================

def coase_bargaining(
    polluter_benefit: Callable[[float], float],
    victim_damage: Callable[[float], float],
    max_pollution: float,
    property_rights: str = 'polluter'
) -> CoaseResult:
    """
    Analyze Coase theorem bargaining outcome.
    
    The Coase theorem states that regardless of initial property rights,
    bargaining will lead to an efficient outcome (in absence of transaction costs).
    
    Parameters:
        polluter_benefit: Function giving polluter's benefit from pollution level Q
        victim_damage: Function giving victim's damage from pollution level Q
        max_pollution: Maximum possible pollution level
        property_rights: 'polluter' or 'victim'
        
    Returns:
        CoaseResult with bargaining outcome
    """
    # Find efficient pollution level (maximize total surplus)
    def total_surplus(Q):
        return polluter_benefit(Q) - victim_damage(Q)
    
    def neg_surplus(Q):
        return -total_surplus(Q)
    
    result = minimize_scalar(neg_surplus, bounds=(0, max_pollution), method='bounded')
    Q_efficient = result.x
    
    benefit_at_efficient = polluter_benefit(Q_efficient)
    damage_at_efficient = victim_damage(Q_efficient)
    max_surplus = total_surplus(Q_efficient)
    
    if property_rights == 'polluter':
        # Polluter has right to pollute
        # Status quo: Q = max_pollution
        benefit_status_quo = polluter_benefit(max_pollution)
        damage_status_quo = victim_damage(max_pollution)
        
        # Victim pays polluter to reduce pollution
        # Payment = polluter's lost benefit from reducing Q
        polluter_loss = benefit_status_quo - benefit_at_efficient
        victim_gain = damage_status_quo - damage_at_efficient
        
        # Bargaining range: [polluter_loss, victim_gain]
        # Split the surplus
        surplus_to_split = victim_gain - polluter_loss
        payment = polluter_loss + 0.5 * surplus_to_split  # Nash bargaining
        
        polluter_payoff = benefit_at_efficient + payment
        victim_payoff = -damage_at_efficient - payment
        
        description = (f"Polluter has property rights. "
                      f"Victim pays ${payment:.2f} to reduce pollution from "
                      f"{max_pollution:.1f} to {Q_efficient:.1f}")
        
    else:  # victim has rights
        # Victim has right to zero pollution
        # Status quo: Q = 0
        benefit_status_quo = polluter_benefit(0)
        damage_status_quo = victim_damage(0)
        
        # Polluter pays victim for permission to pollute
        # Payment = victim's damage from pollution
        victim_loss = damage_at_efficient - damage_status_quo  # = damage_at_efficient
        polluter_gain = benefit_at_efficient - benefit_status_quo
        
        # Bargaining range: [victim_loss, polluter_gain]
        surplus_to_split = polluter_gain - victim_loss
        payment = victim_loss + 0.5 * surplus_to_split  # Nash bargaining
        
        polluter_payoff = benefit_at_efficient - payment
        victim_payoff = -damage_at_efficient + payment
        
        description = (f"Victim has property rights. "
                      f"Polluter pays ${payment:.2f} for permission to pollute "
                      f"at level {Q_efficient:.1f}")
    
    return CoaseResult(
        efficient_quantity=Q_efficient,
        payment=payment,
        polluter_payoff=polluter_payoff,
        victim_payoff=victim_payoff,
        total_surplus=max_surplus,
        property_rights_holder=property_rights,
        description=description
    )


def coase_with_transaction_costs(
    polluter_benefit: Callable[[float], float],
    victim_damage: Callable[[float], float],
    max_pollution: float,
    transaction_cost: float,
    property_rights: str = 'polluter'
) -> Dict:
    """
    Analyze Coase bargaining with transaction costs.
    
    Transaction costs may prevent efficient bargaining if:
    - Gains from trade < transaction costs
    
    Returns:
        Dict with outcome and whether bargaining occurs
    """
    # Efficient outcome without transaction costs
    coase = coase_bargaining(polluter_benefit, victim_damage, max_pollution, property_rights)
    
    if property_rights == 'polluter':
        # Status quo: max pollution
        status_quo_surplus = polluter_benefit(max_pollution) - victim_damage(max_pollution)
        potential_gain = coase.total_surplus - status_quo_surplus
    else:
        # Status quo: zero pollution
        status_quo_surplus = polluter_benefit(0) - victim_damage(0)
        potential_gain = coase.total_surplus - status_quo_surplus
    
    bargaining_occurs = potential_gain > transaction_cost
    
    if bargaining_occurs:
        final_Q = coase.efficient_quantity
        final_surplus = coase.total_surplus - transaction_cost
        description = (f"Bargaining occurs despite ${transaction_cost:.2f} transaction cost. "
                      f"Net gain: ${potential_gain - transaction_cost:.2f}")
    else:
        if property_rights == 'polluter':
            final_Q = max_pollution
        else:
            final_Q = 0
        final_surplus = status_quo_surplus
        description = (f"Transaction cost (${transaction_cost:.2f}) exceeds potential gain "
                      f"(${potential_gain:.2f}). No bargaining occurs.")
    
    return {
        'bargaining_occurs': bargaining_occurs,
        'final_quantity': final_Q,
        'final_surplus': final_surplus,
        'potential_gain': potential_gain,
        'transaction_cost': transaction_cost,
        'efficient_quantity': coase.efficient_quantity,
        'description': description
    }


# ============================================================================
# CAP AND TRADE
# ============================================================================

def cap_and_trade(
    initial_emissions: List[float],
    marginal_abatement_costs: List[Callable[[float], float]],
    cap: float,
    initial_permits: List[float] = None
) -> CapAndTradeResult:
    """
    Analyze cap-and-trade market equilibrium.
    
    Efficient allocation equates marginal abatement costs across firms.
    
    Parameters:
        initial_emissions: Baseline emissions for each firm
        marginal_abatement_costs: MAC functions for each firm (MAC(abatement))
        cap: Total emissions cap
        initial_permits: Initial permit allocation (default: proportional)
        
    Returns:
        CapAndTradeResult with equilibrium
    """
    n_firms = len(initial_emissions)
    total_baseline = sum(initial_emissions)
    required_reduction = total_baseline - cap
    
    if initial_permits is None:
        # Proportional allocation
        initial_permits = [cap * e / total_baseline for e in initial_emissions]
    
    # Find equilibrium permit price where total abatement = required reduction
    # Each firm abates until MAC = permit price
    
    def total_abatement_at_price(price):
        """Total abatement when permit price = price."""
        total = 0
        for i, mac in enumerate(marginal_abatement_costs):
            # Find abatement where MAC(a) = price
            # Abate while MAC < price
            max_abate = initial_emissions[i]
            
            # Binary search for abatement level
            if mac(0) >= price:
                a = 0  # Don't abate
            elif mac(max_abate) <= price:
                a = max_abate  # Abate fully
            else:
                # Find where MAC = price
                try:
                    a = brentq(lambda x: mac(x) - price, 0, max_abate)
                except:
                    a = 0
            total += a
        return total
    
    # Find equilibrium price
    try:
        # Price must be high enough to induce required reduction
        permit_price = brentq(
            lambda p: total_abatement_at_price(p) - required_reduction,
            0, 1000
        )
    except:
        permit_price = 0
    
    # Calculate equilibrium allocations
    firm_abatement = []
    firm_emissions = []
    firm_permit_trades = []
    total_cost = 0
    
    for i, (baseline, mac, permits) in enumerate(zip(
        initial_emissions, marginal_abatement_costs, initial_permits
    )):
        # Abatement at equilibrium price
        if mac(0) >= permit_price:
            a = 0
        elif mac(baseline) <= permit_price:
            a = baseline
        else:
            try:
                a = brentq(lambda x: mac(x) - permit_price, 0, baseline)
            except:
                a = 0
        
        emissions = baseline - a
        trades = emissions - permits  # Positive = buying permits
        
        # Abatement cost = integral of MAC
        from scipy.integrate import quad
        cost, _ = quad(mac, 0, a)
        total_cost += cost
        
        firm_abatement.append(a)
        firm_emissions.append(emissions)
        firm_permit_trades.append(trades)
    
    # Compare to uniform reduction
    uniform_reduction = required_reduction / n_firms
    uniform_cost = 0
    for i, mac in enumerate(marginal_abatement_costs):
        a = min(uniform_reduction, initial_emissions[i])
        cost, _ = quad(mac, 0, a)
        uniform_cost += cost
    
    cost_savings = uniform_cost - total_cost
    
    return CapAndTradeResult(
        cap=cap,
        permit_price=permit_price,
        firm_emissions=firm_emissions,
        firm_abatement=firm_abatement,
        firm_permit_trades=firm_permit_trades,
        total_cost=total_cost,
        cost_savings_vs_uniform=cost_savings
    )


def compare_pollution_policies(
    market: ExternalityMarket,
    target_reduction_pct: float = 0.5
) -> Dict:
    """
    Compare different pollution control policies.
    
    Policies compared:
    1. No intervention (market equilibrium)
    2. Pigouvian tax (optimal)
    3. Quantity standard (command and control)
    4. Cap-and-trade (equivalent to optimal tax)
    
    Parameters:
        market: ExternalityMarket with negative externality
        target_reduction_pct: Target reduction as fraction of market overproduction
        
    Returns:
        Dict comparing policy outcomes
    """
    outcome = market.analyze()
    Q_m = outcome.quantity_market
    Q_opt = outcome.quantity_optimal
    
    # Pigouvian tax
    pigouvian = pigouvian_tax(market)
    
    # Quantity standard (command and control)
    Q_standard = Q_opt
    P_standard = market.demand(Q_standard)
    
    # Welfare under quantity standard (same as optimal if set correctly)
    CS_std = 0.5 * (market.a - P_standard) * Q_standard
    PS_std = 0.5 * (P_standard - market.c) * Q_standard
    ext_cost_std = market.total_external_cost(Q_standard)
    W_standard = CS_std + PS_std - ext_cost_std
    
    return {
        'no_intervention': {
            'quantity': Q_m,
            'welfare': outcome.welfare_market,
            'description': 'Market equilibrium ignoring externality'
        },
        'pigouvian_tax': {
            'quantity': pigouvian.quantity_after_tax,
            'tax_rate': pigouvian.tax_rate,
            'welfare': outcome.welfare_optimal,
            'revenue': pigouvian.tax_revenue,
            'description': 'Optimal corrective tax'
        },
        'quantity_standard': {
            'quantity': Q_standard,
            'welfare': W_standard,
            'description': 'Direct quantity regulation at optimal level'
        },
        'optimal_quantity': Q_opt,
        'welfare_gain_from_intervention': outcome.welfare_optimal - outcome.welfare_market
    }


# ============================================================================
# COMMON EXTERNALITY EXAMPLES
# ============================================================================

def pollution_example(
    demand_intercept: float = 100,
    demand_slope: float = 1,
    mc_intercept: float = 10,
    mc_slope: float = 1,
    pollution_damage: float = 20
) -> Dict:
    """
    Standard pollution externality example.
    
    Returns complete analysis including market outcome, social optimum,
    Pigouvian tax, and welfare comparison.
    """
    market = ExternalityMarket(
        demand_intercept=demand_intercept,
        demand_slope=demand_slope,
        supply_intercept=mc_intercept,
        supply_slope=mc_slope,
        external_cost=pollution_damage,
        external_cost_type='constant',
        externality_type='negative'
    )
    
    outcome = market.analyze()
    tax = pigouvian_tax(market)
    
    return {
        'market': market,
        'outcome': outcome,
        'pigouvian_tax': tax,
        'summary': (
            f"Pollution externality analysis:\n"
            f"  Market produces Q={outcome.quantity_market:.1f} (too much)\n"
            f"  Optimal is Q={outcome.quantity_optimal:.1f}\n"
            f"  Pigouvian tax of ${tax.tax_rate:.2f}/unit corrects inefficiency\n"
            f"  Welfare gain from intervention: ${outcome.welfare_optimal - outcome.welfare_market:.2f}"
        )
    }


def education_spillover_example(
    demand_intercept: float = 80,
    demand_slope: float = 1,
    mc_intercept: float = 20,
    mc_slope: float = 0.5,
    spillover_benefit: float = 15
) -> Dict:
    """
    Positive externality example: education spillovers.
    
    Education creates benefits for society beyond private returns.
    """
    market = ExternalityMarket(
        demand_intercept=demand_intercept,
        demand_slope=demand_slope,
        supply_intercept=mc_intercept,
        supply_slope=mc_slope,
        external_cost=spillover_benefit,
        external_cost_type='constant',
        externality_type='positive'
    )
    
    outcome = market.analyze()
    subsidy = pigouvian_tax(market)
    
    return {
        'market': market,
        'outcome': outcome,
        'pigouvian_subsidy': subsidy,
        'summary': (
            f"Education spillover analysis:\n"
            f"  Market produces Q={outcome.quantity_market:.1f} (too little)\n"
            f"  Optimal is Q={outcome.quantity_optimal:.1f}\n"
            f"  Pigouvian subsidy of ${abs(subsidy.tax_rate):.2f}/unit encourages optimal level\n"
            f"  Welfare gain from intervention: ${outcome.welfare_optimal - outcome.welfare_market:.2f}"
        )
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data classes
    'ExternalityOutcome',
    'PigouvianTaxResult',
    'CoaseResult',
    'CapAndTradeResult',
    
    # Main classes
    'ExternalityMarket',
    
    # Pigouvian analysis
    'pigouvian_tax',
    'tax_incidence_with_externality',
    
    # Coase theorem
    'coase_bargaining',
    'coase_with_transaction_costs',
    
    # Cap and trade
    'cap_and_trade',
    'compare_pollution_policies',
    
    # Examples
    'pollution_example',
    'education_spillover_example',
]
