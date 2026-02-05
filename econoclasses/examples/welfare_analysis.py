"""
Example: Welfare Analysis

Demonstrates:
- First and Second Welfare Theorems
- Social welfare functions (Utilitarian, Rawlsian, Nash)
- Utility Possibility Frontier
- Efficiency vs Equity tradeoff
"""

import sys
sys.path.insert(0, '/home/claude')

from econoclasses import Utility, Consumer, ExchangeEconomy
from econoclasses.welfare import (
    analyze_welfare,
    check_first_welfare_theorem,
    check_second_welfare_theorem,
    utility_possibility_frontier,
    find_welfare_maximizing_allocation,
    utilitarian_welfare,
    rawlsian_welfare,
    nash_welfare,
    gini_coefficient
)
import numpy as np


print("=" * 70)
print("WELFARE ECONOMICS ANALYSIS")
print("=" * 70)

# =============================================================================
# SETUP: EXCHANGE ECONOMY
# =============================================================================

# Two consumers with different preferences and endowments
alice = Consumer(
    "Alice",
    Utility('cobb-douglas', alpha=0.6, beta=0.4),  # Prefers X
    income=0,
    endowment={'X': 8, 'Y': 4}  # Has more X
)

bob = Consumer(
    "Bob",
    Utility('cobb-douglas', alpha=0.3, beta=0.7),  # Prefers Y
    income=0,
    endowment={'X': 4, 'Y': 8}  # Has more Y
)

economy = ExchangeEconomy(alice, bob)

print("\n1. ECONOMY SETUP")
print("-" * 40)
print(f"Alice: U = X^0.6 × Y^0.4, endowment = (8, 4)")
print(f"Bob:   U = X^0.3 × Y^0.7, endowment = (4, 8)")
print(f"Total: X = {economy.total_x}, Y = {economy.total_y}")

# =============================================================================
# WELFARE AT ENDOWMENT
# =============================================================================

print("\n2. WELFARE AT INITIAL ENDOWMENT")
print("-" * 40)

welfare_endow = analyze_welfare(economy, economy.endowment)
print(f"Utilities:")
for name, u in welfare_endow.utilities.items():
    print(f"  {name}: {u:.4f}")

print(f"\nSocial Welfare Measures:")
print(f"  Utilitarian (sum):  {welfare_endow.utilitarian:.4f}")
print(f"  Rawlsian (min):     {welfare_endow.rawlsian:.4f}")
print(f"  Nash (product):     {welfare_endow.nash:.4f}")

utilities_endow = list(welfare_endow.utilities.values())
print(f"  Gini coefficient:   {gini_coefficient(utilities_endow):.4f}")

# =============================================================================
# FIRST WELFARE THEOREM
# =============================================================================

print("\n3. FIRST WELFARE THEOREM")
print("-" * 40)

result1 = check_first_welfare_theorem(economy)
print(result1.explanation)

# =============================================================================
# WELFARE AT EQUILIBRIUM
# =============================================================================

print("\n4. WELFARE AT COMPETITIVE EQUILIBRIUM")
print("-" * 40)

eq = economy.find_equilibrium()
welfare_eq = analyze_welfare(economy, eq.allocation)

print(f"Equilibrium prices: Px = {eq.prices['X']:.4f}, Py = 1.0")
print(f"\nUtilities (equilibrium vs endowment):")
for name in welfare_eq.utilities:
    u_eq = welfare_eq.utilities[name]
    u_end = welfare_endow.utilities[name]
    print(f"  {name}: {u_end:.4f} → {u_eq:.4f}  ({u_eq - u_end:+.4f})")

print(f"\nSocial Welfare Improvement:")
print(f"  Utilitarian: {welfare_endow.utilitarian:.4f} → {welfare_eq.utilitarian:.4f} ({welfare_eq.utilitarian - welfare_endow.utilitarian:+.4f})")
print(f"  Rawlsian:    {welfare_endow.rawlsian:.4f} → {welfare_eq.rawlsian:.4f} ({welfare_eq.rawlsian - welfare_endow.rawlsian:+.4f})")

utilities_eq = list(welfare_eq.utilities.values())
print(f"\nInequality:")
print(f"  Gini: {gini_coefficient(utilities_endow):.4f} → {gini_coefficient(utilities_eq):.4f}")

# =============================================================================
# SECOND WELFARE THEOREM
# =============================================================================

print("\n5. SECOND WELFARE THEOREM")
print("-" * 40)

result2 = check_second_welfare_theorem(economy)
print(result2.explanation)

# =============================================================================
# UTILITY POSSIBILITY FRONTIER
# =============================================================================

print("\n6. UTILITY POSSIBILITY FRONTIER")
print("-" * 40)

u_a, u_b = utility_possibility_frontier(economy, n_points=30)
print(f"Computed {len(u_a)} points on the UPF")
print(f"  Alice utility range: [{min(u_a):.4f}, {max(u_a):.4f}]")
print(f"  Bob utility range:   [{min(u_b):.4f}, {max(u_b):.4f}]")

# =============================================================================
# WELFARE-MAXIMIZING ALLOCATIONS
# =============================================================================

print("\n7. WELFARE-MAXIMIZING ALLOCATIONS")
print("-" * 40)

# Utilitarian optimum
util_alloc, util_w = find_welfare_maximizing_allocation(economy, utilitarian_welfare)
util_analysis = analyze_welfare(economy, util_alloc)

print("Utilitarian (maximize sum):")
print(f"  Alice: ({util_alloc.consumer_a['X']:.2f}, {util_alloc.consumer_a['Y']:.2f}), U = {util_analysis.utilities['Alice']:.4f}")
print(f"  Bob:   ({util_alloc.consumer_b['X']:.2f}, {util_alloc.consumer_b['Y']:.2f}), U = {util_analysis.utilities['Bob']:.4f}")
print(f"  Total welfare: {util_w:.4f}")

# Rawlsian optimum
rawls_alloc, rawls_w = find_welfare_maximizing_allocation(economy, rawlsian_welfare)
rawls_analysis = analyze_welfare(economy, rawls_alloc)

print("\nRawlsian (maximize minimum):")
print(f"  Alice: ({rawls_alloc.consumer_a['X']:.2f}, {rawls_alloc.consumer_a['Y']:.2f}), U = {rawls_analysis.utilities['Alice']:.4f}")
print(f"  Bob:   ({rawls_alloc.consumer_b['X']:.2f}, {rawls_alloc.consumer_b['Y']:.2f}), U = {rawls_analysis.utilities['Bob']:.4f}")
print(f"  Min welfare: {rawls_w:.4f}")

# Nash optimum
nash_alloc, nash_w = find_welfare_maximizing_allocation(economy, nash_welfare)
nash_analysis = analyze_welfare(economy, nash_alloc)

print("\nNash (maximize product):")
print(f"  Alice: ({nash_alloc.consumer_a['X']:.2f}, {nash_alloc.consumer_a['Y']:.2f}), U = {nash_analysis.utilities['Alice']:.4f}")
print(f"  Bob:   ({nash_alloc.consumer_b['X']:.2f}, {nash_alloc.consumer_b['Y']:.2f}), U = {nash_analysis.utilities['Bob']:.4f}")
print(f"  Product welfare: {nash_w:.4f}")

# =============================================================================
# EFFICIENCY VS EQUITY
# =============================================================================

print("\n8. EFFICIENCY VS EQUITY TRADEOFF")
print("-" * 40)
print("""
Key Insight: All three welfare-maximizing allocations are on the
contract curve (Pareto efficient), but they differ in distribution.

- Utilitarian: May allow large inequality if it maximizes total
- Rawlsian: Focuses entirely on worst-off, may sacrifice total
- Nash: Balances efficiency and equality

The Second Welfare Theorem tells us we can achieve ANY of these
efficient allocations through competitive markets + transfers.
""")

print("Done!")
