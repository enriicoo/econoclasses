"""
Example: Exchange Economy and Edgeworth Box

Demonstrates:
- Two consumers with different preferences
- Endowment-based budget constraints  
- Walrasian equilibrium
- Contract curve (Pareto efficiency)
- Gains from trade
"""

import sys
sys.path.insert(0, '/home/claude')

from econoclasses import Utility, Consumer
from econoclasses.equilibrium import ExchangeEconomy
from econoclasses.plotting import plot_edgeworth_box
import matplotlib.pyplot as plt


# =============================================================================
# SETUP: TWO CONSUMERS WITH DIFFERENT PREFERENCES
# =============================================================================

print("=" * 70)
print("EXCHANGE ECONOMY EXAMPLE")
print("=" * 70)

# Alice: balanced preferences, starts with lots of X
alice = Consumer(
    name="Alice",
    utility=Utility('cobb-douglas', alpha=0.5, beta=0.5),
    income=0,  # Will be computed from endowment
    endowment={'X': 10, 'Y': 2}
)

# Bob: prefers Y, starts with lots of Y
bob = Consumer(
    name="Bob",
    utility=Utility('cobb-douglas', alpha=0.3, beta=0.7),
    income=0,
    endowment={'X': 2, 'Y': 10}
)

# Create the exchange economy
economy = ExchangeEconomy(alice, bob)

# Print full summary
print(economy.summary())


# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

# Contract curve
print("\nContract Curve (sample points):")
contract = economy.contract_curve(n_points=10)
for i, alloc in enumerate(contract):
    if i % 3 == 0:  # Print every 3rd point
        print(f"  Alice: X={alloc.consumer_a['X']:.2f}, Y={alloc.consumer_a['Y']:.2f}")

# Core bounds
core_min, core_max = economy.core_bounds()
print(f"\nCore bounds (X for Alice): [{core_min:.2f}, {core_max:.2f}]")

# Equilibrium details
eq = economy.find_equilibrium()
print(f"\nEquilibrium check:")
print(f"  Is Pareto efficient? {economy.is_pareto_efficient(eq.allocation)}")
print(f"  Is in core? {economy.is_in_core(eq.allocation)}")


# =============================================================================
# DIFFERENT PREFERENCES EXAMPLE
# =============================================================================

print("\n" + "=" * 70)
print("ALTERNATIVE: COMPLEMENTS VS SUBSTITUTES")
print("=" * 70)

# Consumer who sees goods as complements
charlie = Consumer(
    name="Charlie",
    utility=Utility('perfect-complements', alpha=1, beta=1),
    income=0,
    endowment={'X': 8, 'Y': 4}
)

# Consumer who sees goods as substitutes
diana = Consumer(
    name="Diana", 
    utility=Utility('cobb-douglas', alpha=0.4, beta=0.6),
    income=0,
    endowment={'X': 4, 'Y': 8}
)

economy2 = ExchangeEconomy(charlie, diana)
eq2 = economy2.find_equilibrium()

print(f"\nCharlie (complements) + Diana (Cobb-Douglas):")
print(f"  Total resources: X={economy2.total_x}, Y={economy2.total_y}")
print(f"  Equilibrium price ratio: {eq2.prices['X']:.4f}")
print(f"  Charlie's allocation: X={eq2.allocation.consumer_a['X']:.2f}, Y={eq2.allocation.consumer_a['Y']:.2f}")
print(f"  Diana's allocation: X={eq2.allocation.consumer_b['X']:.2f}, Y={eq2.allocation.consumer_b['Y']:.2f}")

gains = economy2.gains_from_trade()
print(f"\nGains from trade:")
print(f"  Charlie: {gains['Charlie']:+.4f}")
print(f"  Diana: {gains['Diana']:+.4f}")


# =============================================================================
# PLOTS
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Plot 1: Alice and Bob
fig1, ax1 = plt.subplots(figsize=(10, 8))
plot_edgeworth_box(economy, ax=ax1, title="Alice (α=0.5) vs Bob (α=0.3)")
fig1.tight_layout()
fig1.savefig('/home/claude/econoclasses/examples/edgeworth_alice_bob.png', dpi=150)
print("  → Saved: edgeworth_alice_bob.png")

# Plot 2: Charlie and Diana
fig2, ax2 = plt.subplots(figsize=(10, 8))
plot_edgeworth_box(economy2, ax=ax2, title="Charlie (Complements) vs Diana (CD)")
fig2.tight_layout()
fig2.savefig('/home/claude/econoclasses/examples/edgeworth_charlie_diana.png', dpi=150)
print("  → Saved: edgeworth_charlie_diana.png")

print("\nDone!")
plt.show()
