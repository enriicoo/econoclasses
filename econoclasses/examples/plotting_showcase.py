"""
Plotting Showcase: All econoclasses visualizations

Demonstrates the strategic curve selection system where:
- Curves are always centered on the optimal/equilibrium point
- The middle curve passes exactly through the optimum
- For Edgeworth box: curves meeting at equilibrium are always visible
"""

import matplotlib.pyplot as plt

from econoclasses import (
    Utility, Consumer, Market,
    ProductionFunction,
    ExchangeEconomy, PartialEquilibrium, RobinsonCrusoe
)
from econoclasses.plotting import (
    PlotStyle,
    plot_indifference_curves,
    plot_slutsky_decomposition,
    plot_market_demand,
    plot_market_preferences,
    plot_edgeworth_box,
    plot_supply_demand,
    plot_isoquants,
    plot_robinson_crusoe,
)


def main():
    print("=" * 70)
    print("ECONOCLASSES PLOTTING SHOWCASE")
    print("=" * 70)

    # =========================================================================
    # 1. INDIFFERENCE CURVES (preferences.py)
    # =========================================================================
    print("\n[1/8] Indifference Curves - Cobb-Douglas with equation")

    utility_cd = Utility('cobb-douglas', alpha=0.6, beta=0.4, income=100, price_x=2, price_y=5)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_indifference_curves(
        utility_cd, ax=ax,
        show_budget=True,
        show_optimal=True,
        show_equation=True,  # LaTeX equation in title
        style=PlotStyle(num_curves=5, show_colorbar=True, background_alpha=1.0),
        label_curves=True,
        label_position='last',
        label_direction='horizontal',
    )
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 2. INDIFFERENCE CURVES - Different utility types
    # =========================================================================
    print("\n[2/8] Indifference Curves - Multiple utility types")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    utilities = [
        ('Cobb-Douglas (a=0.5)', Utility('cobb-douglas', alpha=0.5, beta=0.5, income=100, price_x=2, price_y=3),
         True, 'horizontal'),
        ('CES (rho=0.5)', Utility('ces', alpha=0.5, beta=0.5, rho=0.5, income=100, price_x=2, price_y=3),
         'center', 'horizontal'),
        ('Quasilinear', Utility('quasilinear', alpha=0.5, income=100, price_x=2, price_y=3),
         'interpolate', 'diagonal'),
        ('Perfect Substitutes', Utility('perfect-substitutes', alpha=0.6, beta=0.4, income=100, price_x=2, price_y=3),
         'interpolate', 'diagonal'),
    ]

    for ax, (name, util, lcur, ldir) in zip(axes.flat, utilities):
        plot_indifference_curves(
            util, ax=ax,
            show_budget=True,
            show_optimal=True,
            style=PlotStyle(num_curves=5, background_alpha=1.0),
            title=name,
            label_curves=lcur,
            label_position='last',
            label_direction=ldir,
        )

    fig.suptitle("Strategic Curve Selection Across Utility Types", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 3. SLUTSKY DECOMPOSITION
    # =========================================================================
    print("\n[3/8] Slutsky Decomposition")

    utility_slutsky = Utility('cobb-douglas', alpha=0.5, beta=0.5, income=100, price_x=2, price_y=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_slutsky_decomposition(utility_slutsky, new_price_x=4, ax=ax)
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 4. MARKET DEMAND (Price-Quantity space)
    # =========================================================================
    print("\n[4/8] Market Demand")

    consumers = [
        Consumer("Alice", Utility('cobb-douglas', alpha=0.6, beta=0.4), income=100),
        Consumer("Bob", Utility('cobb-douglas', alpha=0.4, beta=0.6), income=80),
        Consumer("Charlie", Utility('cobb-douglas', alpha=0.5, beta=0.5), income=150),
    ]
    market = Market(consumers)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_market_demand(market, good='X', ax=ax, show_individual=True)
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 4b. AGGREGATE PREFERENCES (Goods space - X,Y)
    # Shows individual ICs merging into market preference
    # =========================================================================
    print("\n[4b/8] Aggregate Preferences - ICs merging into market")

    # Create consumers with different utility types
    consumers_prefs = [
        Consumer("Alice", Utility('perfect-substitutes', alpha=0.6, beta=0.4), income=100),
        Consumer("Bob", Utility('perfect-complements', alpha=0.4, beta=0.6), income=100),
        #Consumer("Charlie", Utility('cobb-douglas', alpha=0.5, beta=0.5), income=100),
    ]
    market_prefs = Market(consumers_prefs)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_market_preferences(market_prefs, prices=(2.0, 1.0), ax=ax)
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 5. EDGEWORTH BOX
    # =========================================================================
    print("\n[5/8] Edgeworth Box - 3 curves each, meeting at equilibrium")

    alice = Consumer(
        name="Alice",
        utility=Utility('cobb-douglas', alpha=0.5, beta=0.5),
        income=0,  # Computed from endowment
        endowment={'X': 10, 'Y': 2}
    )
    bob = Consumer(
        name="Bob",
        utility=Utility('cobb-douglas', alpha=0.3, beta=0.7),
        income=0,  # Computed from endowment
        endowment={'X': 2, 'Y': 10}
    )
    economy = ExchangeEconomy(alice, bob)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_edgeworth_box(
        economy, ax=ax,
        n_ic=3,
    )
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 6. SUPPLY AND DEMAND
    # =========================================================================
    print("\n[6/8] Supply and Demand")

    def demand_func(p):
        return max(0, 100 - 5 * p)

    def supply_func(p):
        return max(0, 2 * p - 10)

    partial_eq = PartialEquilibrium(demand_func, supply_func)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_supply_demand(partial_eq, ax=ax, show_surplus=True)
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 7. ISOQUANTS
    # =========================================================================
    print("\n[7/8] Isoquants - strategic selection")

    tech = ProductionFunction('cobb-douglas', A=1, alpha=0.3, beta=0.7)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_isoquants(
        tech, ax=ax,
        K_range=(0.5, 15),
        L_range=(0.5, 15),
        Q_target=5.0,  # Target output for strategic centering
        num_curves=5,
        show_expansion_path=True,
        show_optimal=True,
        wage=5, rental=10,
        style=PlotStyle(background_alpha=1.0),
        title="Isoquants: 5 curves, Q=5 on middle"
    )
    fig.tight_layout()
    plt.show()

    # =========================================================================
    # 8. ROBINSON CRUSOE
    # =========================================================================
    print("\n[8/8] Robinson Crusoe - IC curves centered on optimum")

    rc = RobinsonCrusoe(
        preferences=Utility('cobb-douglas', alpha=0.5, beta=0.5),
        technology=ProductionFunction('cobb-douglas', A=2, alpha=0.5, beta=0.5),
        total_time=24,
        capital=10
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_robinson_crusoe(rc, ax=ax, n_ic=3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
