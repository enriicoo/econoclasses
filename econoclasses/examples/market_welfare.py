"""
Examples for Market Structures, Externalities, and Public Goods

Demonstrates:
- Monopoly vs competition analysis
- Price discrimination strategies
- Externality correction (Pigouvian taxes)
- Coase theorem bargaining
- Public goods provision problems
- Lindahl equilibrium

Author: econoclasses
Version: 0.4.0
"""

import numpy as np
import matplotlib.pyplot as plt


def example_monopoly_analysis():
    """
    Complete monopoly vs perfect competition analysis.
    """
    print("=" * 60)
    print("MONOPOLY VS PERFECT COMPETITION")
    print("=" * 60)
    
    from econoclasses.market_structures import (
        LinearDemand, LinearCost,
        perfect_competition, monopoly,
        compare_market_structures, welfare_comparison_table
    )
    
    # Create demand and cost
    demand = LinearDemand(a=100, b=1)  # P = 100 - Q
    cost = LinearCost(c=20, F=100)     # TC = 100 + 20Q
    
    print("\nMarket Setup:")
    print(f"  Demand: P = 100 - Q")
    print(f"  Cost: TC = 100 + 20Q (MC = 20)")
    
    # Analyze both structures
    pc = perfect_competition(demand, cost)
    mon = monopoly(demand, cost)
    
    print("\n" + "-" * 40)
    print("PERFECT COMPETITION")
    print("-" * 40)
    print(f"  Price = MC = ${pc.price:.2f}")
    print(f"  Quantity = {pc.quantity:.2f}")
    print(f"  Consumer Surplus = ${pc.consumer_surplus:.2f}")
    print(f"  Producer Surplus = ${pc.producer_surplus:.2f}")
    print(f"  Total Surplus = ${pc.total_surplus:.2f}")
    
    print("\n" + "-" * 40)
    print("MONOPOLY")
    print("-" * 40)
    print(f"  Price = ${mon.price:.2f}")
    print(f"  Quantity = {mon.quantity:.2f}")
    print(f"  Profit = ${mon.firm_profit:.2f}")
    print(f"  Consumer Surplus = ${mon.consumer_surplus:.2f}")
    print(f"  Deadweight Loss = ${mon.deadweight_loss:.2f}")
    
    print("\n" + "-" * 40)
    print("COMPARISON TABLE")
    print("-" * 40)
    print(welfare_comparison_table(demand, cost))
    
    return {'competition': pc, 'monopoly': mon}


def example_price_discrimination():
    """
    Price discrimination analysis across market segments.
    """
    print("\n" + "=" * 60)
    print("PRICE DISCRIMINATION")
    print("=" * 60)
    
    from econoclasses.market_structures import (
        LinearDemand, LinearCost,
        monopoly, first_degree_discrimination, 
        third_degree_discrimination, two_part_tariff
    )
    
    # Base case
    demand = LinearDemand(a=100, b=1)
    cost = LinearCost(c=20, F=0)
    
    # Single-price monopoly
    single = monopoly(demand, cost)
    print("\n1. SINGLE-PRICE MONOPOLY")
    print(f"   Price: ${single.price:.2f}")
    print(f"   Quantity: {single.quantity:.2f}")
    print(f"   Profit: ${single.firm_profit:.2f}")
    
    # First-degree (perfect)
    first = first_degree_discrimination(demand, cost)
    print("\n2. FIRST-DEGREE (Perfect) PD")
    print(f"   Quantity: {first.total_quantity:.2f} (efficient)")
    print(f"   Profit: ${first.profit:.2f} (= entire surplus)")
    print(f"   Consumer Surplus: ${first.consumer_surplus:.2f}")
    
    # Third-degree across segments
    demand_high = LinearDemand(a=120, b=1.5)  # Less elastic
    demand_low = LinearDemand(a=80, b=0.8)    # More elastic
    
    third = third_degree_discrimination(
        [demand_high, demand_low], 
        cost,
        market_names=['Premium', 'Budget']
    )
    print("\n3. THIRD-DEGREE PD (Two Segments)")
    print(f"   {third.description}")
    print(f"   Total Profit: ${third.profit:.2f}")
    
    # Two-part tariff
    tpt = two_part_tariff(demand, cost, num_consumers=100)
    print("\n4. TWO-PART TARIFF")
    print(f"   {tpt['description']}")
    print(f"   Quantity per consumer: {tpt['quantity_per_consumer']:.2f}")
    print(f"   Total Profit: ${tpt['profit']:.2f}")
    
    return {'single': single, 'first': first, 'third': third, 'two_part': tpt}


def example_market_power():
    """
    Market concentration and power analysis.
    """
    print("\n" + "=" * 60)
    print("MARKET POWER ANALYSIS")
    print("=" * 60)
    
    from econoclasses.market_structures import (
        lerner_index, herfindahl_index, 
        concentration_ratio, market_power_analysis
    )
    
    # Example industry data
    prices = [50, 48, 52, 45, 47]
    quantities = [1000, 800, 600, 400, 200]
    marginal_costs = [30, 32, 35, 28, 33]
    
    analysis = market_power_analysis(prices, quantities, marginal_costs)
    
    print("\nIndustry with 5 firms:")
    print(f"  Market shares: {[f'{s*100:.1f}%' for s in analysis['market_shares']]}")
    print(f"\n  Herfindahl-Hirschman Index (HHI): {analysis['hhi']:.0f}")
    print(f"  4-firm concentration ratio (CR4): {analysis['cr4']*100:.1f}%")
    print(f"  Industry Lerner Index: {analysis['industry_lerner']:.3f}")
    print(f"\n  {analysis['interpretation']}")
    
    # Individual firm Lerner indices
    print("\n  Individual Lerner Indices:")
    for i, (l, s) in enumerate(zip(analysis['lerner_indices'], analysis['market_shares'])):
        print(f"    Firm {i+1}: L={l:.3f}, Share={s*100:.1f}%")
    
    return analysis


def example_externality_pollution():
    """
    Pollution externality with Pigouvian tax correction.
    """
    print("\n" + "=" * 60)
    print("EXTERNALITY: POLLUTION")
    print("=" * 60)
    
    from econoclasses.externalities import (
        ExternalityMarket, pigouvian_tax, 
        pollution_example, compare_pollution_policies
    )
    
    # Create pollution market
    market = ExternalityMarket(
        demand_intercept=100,
        demand_slope=1,
        supply_intercept=10,
        supply_slope=1,
        external_cost=20,
        external_cost_type='constant',
        externality_type='negative'
    )
    
    outcome = market.analyze()
    tax = pigouvian_tax(market)
    
    print("\nMarket Setup:")
    print("  Demand: P = 100 - Q")
    print("  Private MC: P = 10 + Q")
    print("  External Cost: $20/unit (constant)")
    
    print("\n" + "-" * 40)
    print("MARKET OUTCOME (No Intervention)")
    print("-" * 40)
    print(f"  Quantity: {outcome.quantity_market:.2f}")
    print(f"  Price: ${outcome.price_market:.2f}")
    print(f"  Social Welfare: ${outcome.welfare_market:.2f}")
    
    print("\n" + "-" * 40)
    print("SOCIAL OPTIMUM")
    print("-" * 40)
    print(f"  Quantity: {outcome.quantity_optimal:.2f}")
    print(f"  Price: ${outcome.price_optimal:.2f}")
    print(f"  Social Welfare: ${outcome.welfare_optimal:.2f}")
    
    print("\n" + "-" * 40)
    print("PIGOUVIAN TAX SOLUTION")
    print("-" * 40)
    print(f"  {tax.description}")
    print(f"  Quantity after tax: {tax.quantity_after_tax:.2f}")
    print(f"  Tax revenue: ${tax.tax_revenue:.2f}")
    print(f"  Welfare gain: ${tax.welfare_gain:.2f}")
    
    print("\n" + "-" * 40)
    print("POLICY COMPARISON")
    print("-" * 40)
    policies = compare_pollution_policies(market)
    for name, info in policies.items():
        if isinstance(info, dict):
            print(f"  {name}: Q={info.get('quantity', 'N/A'):.2f}, W={info.get('welfare', 'N/A'):.2f}")
    
    return {'outcome': outcome, 'tax': tax}


def example_coase_theorem():
    """
    Coase theorem bargaining analysis.
    """
    print("\n" + "=" * 60)
    print("COASE THEOREM")
    print("=" * 60)
    
    from econoclasses.externalities import coase_bargaining, coase_with_transaction_costs
    
    # Define polluter benefit and victim damage
    def polluter_benefit(Q):
        return 80 * Q - 0.4 * Q**2  # Concave benefit
    
    def victim_damage(Q):
        return 0.2 * Q**2  # Convex damage
    
    max_pollution = 100
    
    print("\nSetup:")
    print("  Polluter benefit: B(Q) = 80Q - 0.4Q²")
    print("  Victim damage: D(Q) = 0.2Q²")
    
    # Case 1: Polluter has rights
    result1 = coase_bargaining(polluter_benefit, victim_damage, max_pollution, 'polluter')
    print("\n" + "-" * 40)
    print("CASE 1: Polluter Has Property Rights")
    print("-" * 40)
    print(f"  {result1.description}")
    print(f"  Efficient pollution: {result1.efficient_quantity:.2f}")
    print(f"  Payment to polluter: ${result1.payment:.2f}")
    print(f"  Polluter payoff: ${result1.polluter_payoff:.2f}")
    print(f"  Victim payoff: ${result1.victim_payoff:.2f}")
    
    # Case 2: Victim has rights
    result2 = coase_bargaining(polluter_benefit, victim_damage, max_pollution, 'victim')
    print("\n" + "-" * 40)
    print("CASE 2: Victim Has Property Rights")
    print("-" * 40)
    print(f"  {result2.description}")
    print(f"  Efficient pollution: {result2.efficient_quantity:.2f}")
    print(f"  Payment to victim: ${result2.payment:.2f}")
    
    # With transaction costs
    print("\n" + "-" * 40)
    print("WITH TRANSACTION COSTS ($500)")
    print("-" * 40)
    tc_result = coase_with_transaction_costs(
        polluter_benefit, victim_damage, max_pollution, 500, 'polluter'
    )
    print(f"  {tc_result['description']}")
    print(f"  Final quantity: {tc_result['final_quantity']:.2f}")
    
    return {'polluter_rights': result1, 'victim_rights': result2}


def example_public_goods():
    """
    Public goods provision and free rider problem.
    """
    print("\n" + "=" * 60)
    print("PUBLIC GOODS: FREE RIDER PROBLEM")
    print("=" * 60)
    
    from econoclasses.public_goods import (
        PublicGood, voluntary_provision, lindahl_equilibrium,
        compare_provision_mechanisms, national_defense_example
    )
    
    # Create public good with 3 consumers
    def mb1(G): return max(0, 40 - G)
    def mb2(G): return max(0, 30 - 0.8*G)
    def mb3(G): return max(0, 20 - 0.5*G)
    def mc(G): return 30  # Constant MC
    
    pg = PublicGood(
        marginal_benefits=[mb1, mb2, mb3],
        marginal_cost=mc
    )
    
    print("\nSetup: 3 consumers, 1 public good")
    print("  MB₁ = 40 - G")
    print("  MB₂ = 30 - 0.8G")
    print("  MB₃ = 20 - 0.5G")
    print("  MC = 30 (constant)")
    
    # Optimal quantity (Samuelson condition)
    G_opt = pg.optimal_quantity()
    print(f"\nOptimal quantity (ΣMB = MC): G* = {G_opt:.2f}")
    
    # Voluntary provision
    vol = voluntary_provision(pg)
    print("\n" + "-" * 40)
    print("VOLUNTARY PROVISION")
    print("-" * 40)
    print(f"  Quantity provided: {vol.private_quantity:.2f}")
    print(f"  Optimal quantity: {vol.optimal_quantity:.2f}")
    print(f"  Underprovision: {vol.underprovision:.2f}")
    print(f"  Contributions: {[f'{c:.2f}' for c in vol.individual_contributions]}")
    print(f"  Free rider welfare loss: ${vol.free_rider_loss:.2f}")
    
    # Lindahl equilibrium
    lindahl = lindahl_equilibrium(pg)
    print("\n" + "-" * 40)
    print("LINDAHL EQUILIBRIUM")
    print("-" * 40)
    print(f"  Quantity: {lindahl.quantity:.2f}")
    print(f"  Personalized prices: {[f'${p:.2f}' for p in lindahl.personalized_prices]}")
    print(f"  Sum of prices: ${sum(lindahl.personalized_prices):.2f} = MC")
    print(f"  Individual payments: {[f'${p:.2f}' for p in lindahl.individual_payments]}")
    
    return {'public_good': pg, 'voluntary': vol, 'lindahl': lindahl}


def example_contribution_game():
    """
    Public goods contribution game (experimental economics).
    """
    print("\n" + "=" * 60)
    print("PUBLIC GOODS GAME (Experimental)")
    print("=" * 60)
    
    from econoclasses.public_goods import (
        contribution_game, public_goods_game_experiment
    )
    
    # Standard parameters
    n = 4
    endowment = 20
    mpcr = 0.4
    
    print(f"\nGame setup:")
    print(f"  {n} players, ${endowment} endowment each")
    print(f"  MPCR = {mpcr} (marginal per-capita return)")
    print(f"  Payoff: π = (w - c) + MPCR × Σc")
    
    game = contribution_game(n, endowment, mpcr)
    
    print("\n" + "-" * 40)
    print("NASH EQUILIBRIUM")
    print("-" * 40)
    print(f"  Contributions: {game.nash_contributions}")
    print(f"  Total provision: {game.total_provision}")
    print(f"  Individual payoffs: {[f'${p:.2f}' for p in game.individual_payoffs]}")
    
    print("\n" + "-" * 40)
    print("EFFICIENT OUTCOME")
    print("-" * 40)
    print(f"  Would require: everyone contributes ${endowment}")
    print(f"  Efficient provision: {game.efficient_provision}")
    efficient_payoff = endowment - endowment + mpcr * n * endowment
    print(f"  Each would earn: ${efficient_payoff:.2f}")
    
    print("\n" + "-" * 40)
    print("FREE RIDER PROBLEM")
    print("-" * 40)
    print(f"  Efficiency ratio: {game.efficiency_ratio*100:.1f}%")
    print(f"  MPCR < 1 → individual incentive to free ride")
    print(f"  n×MPCR = {n*mpcr} > 1 → social incentive to contribute")
    
    return game


def run_all_examples():
    """Run all market structure and welfare examples."""
    results = {}
    
    results['monopoly'] = example_monopoly_analysis()
    results['price_discrimination'] = example_price_discrimination()
    results['market_power'] = example_market_power()
    results['pollution'] = example_externality_pollution()
    results['coase'] = example_coase_theorem()
    results['public_goods'] = example_public_goods()
    results['contribution_game'] = example_contribution_game()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    run_all_examples()
