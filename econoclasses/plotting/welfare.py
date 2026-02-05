"""
Plotting for welfare analysis.

Utility Possibility Frontier, social indifference curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Callable, List

from ..equilibrium.exchange import ExchangeEconomy
from ..welfare import (
    utility_possibility_frontier,
    analyze_welfare,
    utilitarian_welfare,
    rawlsian_welfare,
    nash_welfare,
    find_welfare_maximizing_allocation
)
from .preferences import strategic_curve_levels


def plot_upf(
    economy: ExchangeEconomy,
    ax: Optional[plt.Axes] = None,
    n_points: int = 50,
    show_endowment: bool = True,
    show_equilibrium: bool = True,
    show_welfare_optima: bool = False,
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot Utility Possibility Frontier.
    
    Shows tradeoff between consumers' utilities on the Pareto frontier.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Compute UPF
    u_a, u_b = utility_possibility_frontier(economy, n_points)
    
    # Sort for plotting
    sorted_idx = np.argsort(u_a)
    u_a = u_a[sorted_idx]
    u_b = u_b[sorted_idx]
    
    # Plot UPF
    ax.plot(u_a, u_b, 'b-', linewidth=2.5, label='UPF (Pareto Frontier)')
    ax.fill_between(u_a, 0, u_b, alpha=0.1, color='blue')
    
    # Endowment point
    if show_endowment:
        w_endow = analyze_welfare(economy, economy.endowment)
        ua_endow = w_endow.utilities[economy.consumer_a.name]
        ub_endow = w_endow.utilities[economy.consumer_b.name]
        ax.plot(ua_endow, ub_endow, 'ko', markersize=10,
                label=f'Endowment ({ua_endow:.2f}, {ub_endow:.2f})')
    
    # Equilibrium point
    if show_equilibrium:
        eq = economy.find_equilibrium()
        w_eq = analyze_welfare(economy, eq.allocation)
        ua_eq = w_eq.utilities[economy.consumer_a.name]
        ub_eq = w_eq.utilities[economy.consumer_b.name]
        ax.plot(ua_eq, ub_eq, 'g*', markersize=15, markeredgecolor='black',
                label=f'Equilibrium ({ua_eq:.2f}, {ub_eq:.2f})')
    
    # Welfare optima
    if show_welfare_optima:
        # Utilitarian
        util_alloc, _ = find_welfare_maximizing_allocation(economy, utilitarian_welfare, n_points)
        w_util = analyze_welfare(economy, util_alloc)
        ax.plot(w_util.utilities[economy.consumer_a.name],
                w_util.utilities[economy.consumer_b.name],
                'r^', markersize=10, label='Utilitarian')
        
        # Rawlsian
        rawls_alloc, _ = find_welfare_maximizing_allocation(economy, rawlsian_welfare, n_points)
        w_rawls = analyze_welfare(economy, rawls_alloc)
        ax.plot(w_rawls.utilities[economy.consumer_a.name],
                w_rawls.utilities[economy.consumer_b.name],
                'ms', markersize=10, label='Rawlsian')
        
        # 45-degree line (equal utility)
        max_u = max(max(u_a), max(u_b))
        ax.plot([0, max_u], [0, max_u], 'k:', alpha=0.5, label='Equal utility')
    
    ax.set_xlabel(f'Utility of {economy.consumer_a.name}', fontsize=11)
    ax.set_ylabel(f'Utility of {economy.consumer_b.name}', fontsize=11)
    ax.set_xlim(0, max(u_a) * 1.1)
    ax.set_ylim(0, max(u_b) * 1.1)
    ax.set_title(title or 'Utility Possibility Frontier')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_social_indifference_curves(
    ax: plt.Axes,
    welfare_func: Callable[[List[float]], float],
    u_a_range: Tuple[float, float],
    u_b_range: Tuple[float, float],
    levels: Optional[List[float]] = None,
    optimal_welfare: Optional[float] = None,
    color: str = 'red',
    n_levels: int = 5,
) -> None:
    """
    Add social indifference curves to an existing plot.

    Social IC: combinations of (U_A, U_B) giving same social welfare.

    Parameters
    ----------
    optimal_welfare : float, optional
        If provided, curves are strategically selected so this value is at the middle.
    """
    u_a = np.linspace(u_a_range[0], u_a_range[1], 100)
    u_b = np.linspace(u_b_range[0], u_b_range[1], 100)
    U_A, U_B = np.meshgrid(u_a, u_b)

    # Compute social welfare at each point
    W = np.zeros_like(U_A)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            try:
                W[i, j] = welfare_func([U_A[i, j], U_B[i, j]])
            except Exception:
                W[i, j] = np.nan

    # Auto-select levels strategically if optimal_welfare provided
    if levels is None:
        if optimal_welfare is not None:
            levels = strategic_curve_levels(optimal_welfare, n_levels, spread=0.4)
        else:
            w_min = np.nanmin(W[W > 0])
            w_max = np.nanmax(W)
            levels = np.linspace(w_min, w_max, n_levels)

    ax.contour(U_A, U_B, W, levels=levels, colors=[color], alpha=0.5, linewidths=1)


