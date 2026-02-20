"""
Example: Robinson Crusoe Economy

Demonstrates:
- Single agent as both consumer and producer
- Production Possibility Frontier
- Optimal labor-leisure choice
- Decentralization (First Welfare Theorem in action)
"""


from econoclasses import Utility, ProductionFunction, RobinsonCrusoe
from econoclasses.plotting import plot_robinson_crusoe, plot_ppf
import matplotlib.pyplot as plt


print("=" * 70)
print("ROBINSON CRUSOE ECONOMY")
print("=" * 70)

# =============================================================================
# SETUP
# =============================================================================

# Robinson's preferences over consumption (C) and leisure (R)
# U(C, R) = C^0.4 × R^0.6  (values leisure somewhat more than consumption)
preferences = Utility('cobb-douglas', alpha=0.4, beta=0.6, 
                      income=100, price_x=1, price_y=1)

# Production technology: coconuts from labor
# Q = 10 × K^0.3 × L^0.7  (labor-intensive)
technology = ProductionFunction('cobb-douglas', alpha=0.3, A=10)

# Robinson has 16 waking hours and 1 unit of capital (his tools)
rc = RobinsonCrusoe(preferences, technology, total_time=16, capital=1)

# =============================================================================
# FIND OPTIMAL ALLOCATION
# =============================================================================

print("\n1. CENTRAL PLANNER SOLUTION")
print("-" * 40)
print("Maximize U(C, R) subject to:")
print("  C = f(K, L)  [consumption = production]")
print("  R = T - L    [leisure = time - labor]")
print()

eq = rc.find_equilibrium()

print(f"Optimal allocation:")
print(f"  Labor (L*):      {eq.labor:.2f} hours")
print(f"  Leisure (R*):    {eq.leisure:.2f} hours")
print(f"  Output (Q* = C*): {eq.output:.2f} coconuts")
print(f"  Utility (U*):    {eq.utility:.4f}")

# =============================================================================
# DECENTRALIZED EQUILIBRIUM
# =============================================================================

print("\n2. DECENTRALIZED INTERPRETATION")
print("-" * 40)
print("Same allocation achieved through markets:")
print()

print(f"Shadow wage (w*): {eq.wage:.4f} coconuts/hour")
print(f"  = Marginal Product of Labor at L*")
print()

verify = rc.verify_decentralization(eq)

print("Verification:")
print(f"  Firm FOC (P×MPL = w): {verify['firm_foc_satisfied']}")
print(f"    Value: P×MPL = {verify['firm_foc_value']:.4f}, w = {eq.wage:.4f}")
print()
print(f"  Consumer FOC (MRS = w/P): {verify['consumer_foc_satisfied']}")
print(f"    MRS = {verify['mrs']:.4f}, w/P = {verify['wage_price_ratio']:.4f}")

# =============================================================================
# FIRST WELFARE THEOREM
# =============================================================================

print("\n3. FIRST WELFARE THEOREM")
print("-" * 40)
print("""
The theorem states: Competitive equilibrium → Pareto efficient.

In Robinson Crusoe:
- The central planner's solution IS the Pareto efficient allocation
  (there's only one agent, so any utility-maximizing choice is efficient)
  
- The decentralized equilibrium achieves the SAME allocation
  
- Therefore, the competitive equilibrium is Pareto efficient ✓

This illustrates why markets can achieve efficiency:
the price (wage) signals the true opportunity cost of leisure.
""")

# =============================================================================
# COMPARATIVE STATICS
# =============================================================================

print("\n4. WHAT IF TECHNOLOGY IMPROVES?")
print("-" * 40)

# Better technology: A increases from 10 to 15
better_tech = ProductionFunction('cobb-douglas', alpha=0.3, A=15)
rc_better = RobinsonCrusoe(preferences, better_tech, total_time=16, capital=1)
eq_better = rc_better.find_equilibrium()

print(f"With A = 15 (50% productivity increase):")
print(f"  Labor:   {eq.labor:.2f} → {eq_better.labor:.2f} ({eq_better.labor - eq.labor:+.2f})")
print(f"  Leisure: {eq.leisure:.2f} → {eq_better.leisure:.2f} ({eq_better.leisure - eq.leisure:+.2f})")
print(f"  Output:  {eq.output:.2f} → {eq_better.output:.2f} ({eq_better.output - eq.output:+.2f})")
print(f"  Utility: {eq.utility:.2f} → {eq_better.utility:.2f} ({eq_better.utility - eq.utility:+.2f})")
print()
print("Robinson works less but produces and consumes more!")
print("Higher productivity → can afford more leisure AND more consumption.")

# =============================================================================
# PLOTS
# =============================================================================

print("\n5. GENERATING PLOTS")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Full Robinson Crusoe diagram
plot_robinson_crusoe(rc, ax=axes[0], title='Robinson Crusoe: PPF and Indifference Curves')

# Plot 2: PPF alone with tangent
plot_ppf(rc, ax=axes[1])

plt.tight_layout()
plt.show()

print("\nDone!")
