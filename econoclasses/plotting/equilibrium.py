"""
Plotting for equilibrium analysis.

Includes Edgeworth box and supply/demand diagrams.

Key conventions:
- Edgeworth: 3 curves per consumer, red=before equilibrium, blue=after
- Supply-demand: equilibrium centered in plot (both X and Y)
- Colormap support
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..equilibrium import ExchangeEconomy, PartialEquilibrium
from .preferences import strategic_curve_levels


def plot_edgeworth_box(
    economy: ExchangeEconomy,
    ax: Optional[plt.Axes] = None,
    show_contract_curve: bool = True,
    show_equilibrium: bool = True,
    show_indifference_curves: bool = True,
    n_ic: int = 3,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot Edgeworth box for two-consumer exchange economy.

    Key features:
    - Strategic curve selection: middle curve passes through equilibrium
    - Color scheme: black (box, contract curve), gray (ICs), red (equilibrium)
    - Consumer A: solid gray curves
    - Consumer B: dashed gray curves

    Parameters
    ----------
    economy : ExchangeEconomy
        The exchange economy to visualize.
    n_ic : int
        Number of indifference curves per consumer (default 5).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    total_x = economy.total_x
    total_y = economy.total_y

    # Draw the box
    ax.plot([0, total_x, total_x, 0, 0], [0, 0, total_y, total_y, 0], 'k-', linewidth=2)

    # Grid for contours
    x = np.linspace(0.01, total_x - 0.01, 200)
    y = np.linspace(0.01, total_y - 0.01, 200)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Find equilibrium first to center curves on it
    eq = economy.find_equilibrium() if (show_equilibrium or show_indifference_curves) else None

    if show_indifference_curves:
        # Consumer A's utility grid (origin bottom-left)
        U_a = economy.consumer_a.utility._utility_func(X_grid, Y_grid)
        U_a = np.where(np.isfinite(U_a), U_a, np.nan)

        # Consumer B's utility grid (origin top-right, transform coordinates)
        X_b = total_x - X_grid
        Y_b = total_y - Y_grid
        U_b = economy.consumer_b.utility._utility_func(X_b, Y_b)
        U_b = np.where(np.isfinite(U_b), U_b, np.nan)

        # Get equilibrium utilities
        if eq is not None and eq.converged:
            eq_x = eq.allocation.consumer_a['X']
            eq_y = eq.allocation.consumer_a['Y']
            u_a_eq = economy.consumer_a.utility.utility_at(eq_x, eq_y)
            u_b_eq = economy.consumer_b.utility.utility_at(total_x - eq_x, total_y - eq_y)
        else:
            # Fall back to endowment if no equilibrium
            u_a_eq = economy.consumer_a.utility.utility_at(
                economy.endowment.consumer_a['X'],
                economy.endowment.consumer_a['Y']
            )
            u_b_eq = economy.consumer_b.utility.utility_at(
                economy.endowment.consumer_b['X'],
                economy.endowment.consumer_b['Y']
            )

        # Curves AT and BELOW equilibrium utility
        # Highest curve (last) passes through equilibrium
        color_a = '#009E73'  # Teal for consumer A
        color_b = '#CC79A7'  # Pink for consumer B

        spread = 0.4
        step_a = spread * u_a_eq / max(n_ic - 1, 1)
        levels_a = [max(u_a_eq - (n_ic - 1 - i) * step_a, 0.01) for i in range(n_ic)]

        step_b = spread * u_b_eq / max(n_ic - 1, 1)
        levels_b = [max(u_b_eq - (n_ic - 1 - i) * step_b, 0.01) for i in range(n_ic)]

        # Consumer A curves (solid, all teal)
        for i, level in enumerate(levels_a):
            is_eq_curve = (i == n_ic - 1)
            lw = 2.5 if is_eq_curve else 1.5
            ax.contour(X_grid, Y_grid, U_a, levels=[level],
                       colors=[color_a], linewidths=lw, linestyles='-')

        # Consumer B curves (solid, all pink)
        for i, level in enumerate(levels_b):
            is_eq_curve = (i == n_ic - 1)
            lw = 2.5 if is_eq_curve else 1.5
            ax.contour(X_grid, Y_grid, U_b, levels=[level],
                       colors=[color_b], linewidths=lw, linestyles='-')

        # Legend entries
        ax.plot([], [], color=color_a, linewidth=1.5, linestyle='-',
                label=f'{economy.consumer_a.name} ICs')
        ax.plot([], [], color=color_b, linewidth=1.5, linestyle='-',
                label=f'{economy.consumer_b.name} ICs')

    # Contract curve
    if show_contract_curve:
        cc_x, cc_y = economy.contract_curve_arrays(n_points=100)
        ax.plot(cc_x, cc_y, 'k-', linewidth=2, label='Contract Curve', alpha=0.6)

    # Endowment point
    endow_x = economy.endowment.consumer_a['X']
    endow_y = economy.endowment.consumer_a['Y']
    ax.plot(endow_x, endow_y, 'ko', markersize=10, markeredgecolor='white',
            markeredgewidth=1.5, label='Endowment', zorder=5)

    # Equilibrium
    if show_equilibrium and eq is not None and eq.converged:
        eq_x = eq.allocation.consumer_a['X']
        eq_y = eq.allocation.consumer_a['Y']
        ax.plot(eq_x, eq_y, 'o', color='indianred', markersize=14,
                markeredgecolor='darkred', markeredgewidth=2,
                label=f'Equilibrium ($P_x/P_y$={eq.prices["X"]:.2f})', zorder=6)

        # Price line through endowment
        slope = -eq.prices['X'] / eq.prices['Y']
        x_line = np.linspace(0, total_x, 100)
        y_line = endow_y + slope * (x_line - endow_x)
        mask = (y_line >= 0) & (y_line <= total_y)
        ax.plot(x_line[mask], y_line[mask], color='indianred', linestyle=':',
                linewidth=1.5, alpha=0.7)

    # Labels for Consumer A (bottom-left origin)
    ax.set_xlabel(f'{economy.consumer_a.name}: Good X $\\rightarrow$', fontsize=11)
    ax.set_ylabel(f'{economy.consumer_a.name}: Good Y $\\rightarrow$', fontsize=11)

    # Secondary axes labels for Consumer B (top-right origin)
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xlabel(f'$\\leftarrow$ {economy.consumer_b.name}: Good X', fontsize=11)
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f'{total_x - t:.1f}' if 0 <= t <= total_x else '' for t in ticks])

    ax3 = ax.secondary_yaxis('right')
    ax3.set_ylabel(f'$\\leftarrow$ {economy.consumer_b.name}: Good Y', fontsize=11)
    ticks_y = ax.get_yticks()
    ax3.set_yticks(ticks_y)
    ax3.set_yticklabels([f'{total_y - t:.1f}' if 0 <= t <= total_y else '' for t in ticks_y])

    ax.set_xlim(0, total_x)
    ax.set_ylim(0, total_y)
    ax.set_title(title or 'Edgeworth Box')
    ax.legend(loc='lower left')
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_supply_demand(
    equilibrium: PartialEquilibrium,
    ax: Optional[plt.Axes] = None,
    price_range: Optional[Tuple[float, float]] = None,
    show_equilibrium: bool = True,
    show_surplus: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot supply and demand curves with equilibrium centered in frame.

    Key features:
    - Equilibrium point at center of plot (both X and Y axes)
    - Price and quantity ranges symmetric around equilibrium
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Find equilibrium first to center the plot
    eq = equilibrium.find_equilibrium()

    # Compute ranges centered on equilibrium
    if not np.isnan(eq.price) and not np.isnan(eq.quantity):
        p_eq = eq.price
        q_eq = eq.quantity

        # Center equilibrium in frame: eq at (0.5, 0.5)
        # Price range: 0 to 2*p_eq puts eq at center
        p_min = max(0.1, p_eq * 0.2)
        p_max = p_eq * 1.8

        # Quantity range: 0 to 2*q_eq puts eq at center
        q_max = q_eq * 2.0

        if price_range is None:
            price_range = (p_min, p_max)
    else:
        if price_range is None:
            price_range = equilibrium.price_bounds
        q_max = None

    prices = np.linspace(price_range[0], price_range[1], 100)
    demand_qty = [max(0, equilibrium.demand(p)) for p in prices]
    supply_qty = [max(0, equilibrium.supply(p)) for p in prices]

    # Determine quantity range
    if q_max is None:
        q_max = max(max(demand_qty), max(supply_qty)) * 1.1

    # Plot curves
    ax.plot(demand_qty, prices, 'b-', linewidth=2.5, label='Demand')
    ax.plot(supply_qty, prices, 'r-', linewidth=2.5, label='Supply')

    if show_equilibrium and not np.isnan(eq.price):
        ax.plot(eq.quantity, eq.price, 'ko', markersize=12,
                markeredgecolor='white', markeredgewidth=2,
                label=f'Eq: P={eq.price:.2f}, Q={eq.quantity:.2f}', zorder=10)
        ax.axhline(eq.price, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(eq.quantity, color='gray', linestyle=':', alpha=0.5)

    if show_surplus and not np.isnan(eq.price):
        # Consumer surplus (area above price, below demand)
        demand_prices = np.linspace(eq.price, price_range[1], 50)
        demand_at_prices = [max(0, equilibrium.demand(p)) for p in demand_prices]
        ax.fill_betweenx(demand_prices, 0, demand_at_prices, alpha=0.3, color='blue',
                         label=f'CS={eq.consumer_surplus:.2f}')

        # Producer surplus (area below price, above supply)
        supply_prices = np.linspace(price_range[0], eq.price, 50)
        supply_at_prices = [max(0, equilibrium.supply(p)) for p in supply_prices]
        ax.fill_betweenx(supply_prices, 0, supply_at_prices, alpha=0.3, color='red',
                         label=f'PS={eq.producer_surplus:.2f}')

    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    ax.set_title(title or 'Supply and Demand')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, q_max)
    ax.set_ylim(price_range[0], price_range[1])

    return ax
