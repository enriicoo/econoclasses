"""
Plotting for production analysis.

Isoquants, cost curves, supply curves.

Key conventions (SHARED with preferences - uses same core logic):
- Optimal point at center of frame (0.5, 0.5 of axes)
- Curves span 0.25-0.75 of the diagonal
- Middle curve passes exactly through the optimal point
- Background uses mid-range colormap (alpha 0.4), curves use extremes (0.6-1.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from ..production import ProductionFunction, Firm, Industry
from .preferences import (
    strategic_curve_levels,
    PlotStyle,
    _compute_centered_frame_generic,
    _plot_contour_core,
)


def plot_isoquants(
    tech: ProductionFunction,
    ax: Optional[plt.Axes] = None,
    K_range: Optional[Tuple[float, float]] = None,
    L_range: Optional[Tuple[float, float]] = None,
    Q_levels: Optional[List[float]] = None,
    show_expansion_path: bool = False,
    show_optimal: bool = False,
    Q_target: Optional[float] = None,
    wage: float = 5,
    rental: float = 10,
    num_curves: int = 5,
    style: Optional[PlotStyle] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot isoquants for a production function.

    Uses the SAME core logic as plot_indifference_curves (shared via _plot_contour_core):
    - Optimal point centered at (0.5, 0.5) of frame
    - Curves span 0.25-0.75 of diagonal
    - Middle curve passes through optimal
    - Background gradient (alpha 0.4), curves at colormap extremes (0.6-1.0)

    Parameters
    ----------
    tech : ProductionFunction
        The production technology.
    Q_levels : list, optional
        Output levels to plot. If None, auto-selected strategically.
    show_expansion_path : bool
        If True, show cost-minimizing K/L ratio as expansion path.
    show_optimal : bool
        If True, show the cost-minimizing point for Q_target.
    Q_target : float, optional
        Target output level for optimal point. If None, uses default.
    num_curves : int
        Number of isoquants to display (default 5, middle one at optimal).
    style : PlotStyle, optional
        Plot styling configuration.
    """
    style = style or PlotStyle()  # Uses viridis by default (matching v0.1.0)

    # If Q_target not specified, compute one for centering
    if Q_target is None:
        Q_target = tech.output_at(5, 5)

    # Get optimal point for centering
    K_star, L_star = None, None
    if show_optimal or K_range is None or L_range is None:
        try:
            sol = tech.cost_minimize(Q_target, wage, rental)
            K_star, L_star = sol.K, sol.L
        except Exception:
            K_star, L_star = 5.0, 5.0

    # Use SAME frame centering logic as indifference curves
    if K_range is None or L_range is None:
        auto_K, auto_L = _compute_centered_frame_generic(K_star, L_star, 20.0, 20.0)
        K_range = K_range or auto_K
        L_range = L_range or auto_L

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Auto-select Q levels strategically centered on Q_target
    if Q_levels is None:
        Q_levels = strategic_curve_levels(Q_target, num_curves, spread=0.5)

    # Use SHARED core plotting logic (same as indifference curves)
    ax, cs, cmap_obj = _plot_contour_core(
        ax=ax,
        value_func=tech._production_func,
        x_range=K_range,
        y_range=L_range,
        levels=Q_levels,
        style=style,
        colorbar_label='Q',
    )

    # Add contour labels for isoquants
    ax.clabel(cs, inline=True, fontsize=8, fmt='Q=%.1f')

    # Expansion path (cost-minimizing K/L combinations)
    if show_expansion_path:
        expansion_K = []
        expansion_L = []
        for Q in np.linspace(min(Q_levels), max(Q_levels), 20):
            try:
                sol = tech.cost_minimize(Q, wage, rental)
                expansion_K.append(sol.K)
                expansion_L.append(sol.L)
            except Exception:
                pass

        if expansion_K:
            # Use inverse color for contrast
            mid_color = cmap_obj(0.5)
            inv_color = (1 - mid_color[0], 1 - mid_color[1], 1 - mid_color[2])
            ax.plot(expansion_K, expansion_L, '--', color=inv_color,
                    linewidth=2, label='Expansion Path')

    # Show optimal cost-minimizing point
    if show_optimal and K_star is not None:
        mid_color = cmap_obj(0.5)
        inv_color = (1 - mid_color[0], 1 - mid_color[1], 1 - mid_color[2])

        ax.plot(K_star, L_star, 'o', color=inv_color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'Optimal (K={K_star:.2f}, L={L_star:.2f})', zorder=10)

    ax.set_xlabel('Capital (K)')
    ax.set_ylabel('Labor (L)')
    ax.set_xlim(K_range)
    ax.set_ylim(L_range)
    ax.set_title(title or f'Isoquants: {tech.form_name}')

    if style.show_grid:
        ax.grid(True, alpha=0.3)

    if show_expansion_path or show_optimal:
        ax.legend(loc='upper right')

    # Info box
    if show_optimal and K_star is not None:
        try:
            sol = tech.cost_minimize(Q_target, wage, rental)
            info_lines = [
                f"$w = {wage:.2f}$  |  $r = {rental:.2f}$",
                f"$K^* = {K_star:.2f}$  |  $L^* = {L_star:.2f}$",
                f"$Q^* = {Q_target:.2f}$  |  $C^* = {sol.cost:.2f}$"
            ]
            info_text = "\n".join(info_lines)
            ax.text(
                0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8)
            )
        except Exception:
            pass

    return ax


def plot_firm_supply(
    firm: Firm,
    ax: Optional[plt.Axes] = None,
    price_range: Tuple[float, float] = (1, 20),
    n_points: int = 50,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot supply curve for a single firm."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    supply = firm.supply_curve(price_range, n_points)

    ax.plot(supply['Q'], supply['P'], 'b-', linewidth=2)

    ax.set_xlabel('Quantity (Q)')
    ax.set_ylabel('Price (P)')
    ax.set_title(title or f'Supply Curve: {firm.name}')
    ax.grid(True, alpha=0.3)

    return ax


def plot_industry_supply(
    industry: Industry,
    ax: Optional[plt.Axes] = None,
    price_range: Tuple[float, float] = (1, 20),
    n_points: int = 50,
    show_individual: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    cmap: str = 'viridis',
) -> plt.Axes:
    """Plot industry supply curve (aggregate of firms)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    supply = industry.supply_curve(price_range, n_points)

    if show_individual:
        cmap_obj = plt.get_cmap(cmap)
        n_firms = len(industry.firms)
        for i, firm in enumerate(industry.firms):
            color = cmap_obj(0.3 + 0.5 * i / max(n_firms - 1, 1))
            ax.plot(supply[firm.name], supply['P'], alpha=0.6, linewidth=1.5,
                    color=color, label=firm.name)

    ax.plot(supply['aggregate'], supply['P'], 'k-', linewidth=2.5, label='Industry')

    ax.set_xlabel('Quantity (Q)')
    ax.set_ylabel('Price (P)')
    ax.set_title(title or 'Industry Supply')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


