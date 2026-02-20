"""
Example: Five Friends Choosing Between Burgers and Pizza

Demonstrates how different preference types lead to different choices,
and how individual demands aggregate into market demand.
"""


from econoclasses import Utility, Consumer, Market

# =============================================================================
# DEFINE THE FIVE FRIENDS
# =============================================================================

print("=" * 70)
print("FIVE FRIENDS: BURGERS VS PIZZA")
print("=" * 70)
print()

# Each friend has different preferences
friends = [
    Consumer(
        name="Alice",
        utility=Utility('cobb-douglas', alpha=0.5, beta=0.5),
        income=200  # Balanced preferences
    ),
    Consumer(
        name="Bob",
        utility=Utility('cobb-douglas', alpha=0.8, beta=0.2),
        income=150  # Burger lover
    ),
    Consumer(
        name="Carol",
        utility=Utility('cobb-douglas', alpha=0.2, beta=0.8),
        income=180  # Pizza fanatic
    ),
    Consumer(
        name="Dave",
        utility=Utility('perfect-complements', alpha=2, beta=1),
        income=200  # Always wants 2 burgers per pizza
    ),
    Consumer(
        name="Eve",
        utility=Utility('perfect-substitutes', alpha=1, beta=1.2),
        income=160  # Price-sensitive, slight preference for pizza
    ),
]

market = Market(friends)

# =============================================================================
# SCENARIO 1: BASELINE PRICES
# =============================================================================

prices = {'X': 8, 'Y': 10}  # Burger = $8, Pizza = $10
print(f"Prices: Burger (X) = ${prices['X']}, Pizza (Y) = ${prices['Y']}")
print()
print(market.summary_table(prices))

# Explain each person's behavior
print("\nWhy each friend behaves this way:")
print("-" * 50)
for c in friends:
    d = c.demand(prices)
    u = c.utility
    
    if u.form_name == 'cobb-douglas':
        share = u._params.get('alpha', 0.5) / (u._params.get('alpha', 0.5) + u._params.get('beta', 0.5))
        print(f"{c.name}: Spends {share*100:.0f}% on burgers (CD preferences)")
    elif u.form_name == 'perfect-complements':
        print(f"{c.name}: Must have 2 burgers per pizza (complements)")
    elif u.form_name == 'perfect-substitutes':
        if d['X'] > 0 and d['Y'] == 0:
            print(f"{c.name}: All burgers - better value per util (substitutes)")
        elif d['Y'] > 0 and d['X'] == 0:
            print(f"{c.name}: All pizza - better value per util (substitutes)")
        else:
            print(f"{c.name}: Indifferent between goods (substitutes)")

# =============================================================================
# SCENARIO 2: PIZZA SALE
# =============================================================================

print("\n" + "=" * 70)
print("SCENARIO 2: PIZZA SALE! (Price drops from $10 to $6)")
print("=" * 70)
print()

new_prices = {'X': 8, 'Y': 6}
print(market.price_change_analysis(prices, new_prices))

# =============================================================================
# SCENARIO 3: SLUTSKY DECOMPOSITION FOR ALICE
# =============================================================================

print("\n" + "=" * 70)
print("SLUTSKY DECOMPOSITION: Alice's Response to Pizza Sale")
print("=" * 70)

alice = friends[0]
alice.utility.price_x = prices['X']
alice.utility.price_y = prices['Y']
alice.utility.income = alice.income

decomp = alice.utility.slutsky_decomposition(new_price_y=6, good='Y')

print(f"""
Alice's pizza consumption when price drops $10 → $6:

Original bundle: X = {decomp['original_bundle'][0]:.2f}, Y = {decomp['original_bundle'][1]:.2f}
Compensated:     X = {decomp['compensated_bundle'][0]:.2f}, Y = {decomp['compensated_bundle'][1]:.2f}
Final bundle:    X = {decomp['new_bundle'][0]:.2f}, Y = {decomp['new_bundle'][1]:.2f}

Substitution effect (ΔY): {decomp['substitution_effect']:+.2f}
  → She substitutes toward cheaper pizza

Income effect (ΔY): {decomp['income_effect']:+.2f}
  → Her purchasing power increased

Total effect (ΔY): {decomp['total_effect']:+.2f}
""")

# =============================================================================
# MARKET DEMAND SCHEDULE
# =============================================================================

print("=" * 70)
print("MARKET DEMAND SCHEDULE (Pizza prices from $4 to $15)")
print("=" * 70)
print()

schedule = market.demand_schedule(good='Y', price_range=(4, 15), other_price=8, n_points=6)

print(f"{'Price':<8}", end='')
for c in friends:
    print(f"{c.name:<10}", end='')
print(f"{'TOTAL':<10}")

print("-" * 68)

for i, p in enumerate(schedule['prices']):
    print(f"${p:<7.2f}", end='')
    for c in friends:
        print(f"{schedule[c.name][i]:<10.2f}", end='')
    print(f"{schedule['aggregate'][i]:<10.2f}")

print("\nNote: As pizza price rises, total demand falls (law of demand).")
print("Eve (substitutes) shows most dramatic response - switches entirely.")
