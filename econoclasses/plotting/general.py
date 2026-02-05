"""
Plotting for general equilibrium models.

Robinson Crusoe PPF + indifference curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..equilibrium.general import RobinsonCrusoe
from .preferences import strategic_curve_levels


def plot_robinson_crusoe(
    rc: RobinsonCrusoe,
    ax: Optional[plt.Axes] = None,
    show_optimum: bool = True,
    show_budget_line: bool = True,
    n_ic: int = 3,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot Robinson Crusoe economy: PPF and indifference curves.

    X-axis: Consumption (C)
    Y-axis: Leisure (R)

    PPF shows feasible (C, R) combinations.
    Optimum is where IC is tangent to PPF.

    Color scheme: black (PPF), gray (ICs), red (optimum)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    eq = rc.find_equilibrium()
    T = rc.total_time

    # Production Possibility Frontier
    R_vals = np.linspace(0.1, T - 0.1, 200)
    C_vals = np.array([rc.output(T - R) for R in R_vals])

    ax.plot(C_vals, R_vals, 'k-', linewidth=2.5, label='PPF')
    ax.fill_between(C_vals, R_vals, alpha=0.08, color='black')

    # Indifference curves
    C_grid = np.linspace(0.1, max(C_vals) * 1.2, 200)
    R_grid = np.linspace(0.1, T, 200)
    C_mesh, R_mesh = np.meshgrid(C_grid, R_grid)

    try:
        U_mesh = rc.preferences._utility_func(C_mesh, R_mesh)
        U_mesh = np.where(np.isfinite(U_mesh), U_mesh, np.nan)

        # ICs AT and BELOW optimal utility (highest curve at optimum)
        u_opt = eq.utility
        spread = 0.4
        step = spread * u_opt / max(n_ic - 1, 1)
        u_levels = [max(u_opt - (n_ic - 1 - i) * step, 0.01) for i in range(n_ic)]

        # All curves teal (#009E73)
        for i, level in enumerate(u_levels):
            is_optimal = (i == n_ic - 1)
            lw = 2.5 if is_optimal else 1.5
            ax.contour(C_mesh, R_mesh, U_mesh, levels=[level],
                       colors=['#009E73'], linewidths=lw)

        # Legend entry for ICs
        ax.plot([], [], color='#009E73', linewidth=1.5, label='Indifference curves')
    except Exception:
        pass

    # Budget line (decentralized interpretation)
    if show_budget_line:
        w = eq.wage
        R_intercept = T
        budget_R = np.linspace(0, R_intercept, 100)
        budget_C = w * (T - budget_R)
        mask = budget_C >= 0
        ax.plot(budget_C[mask], budget_R[mask], color='gray', linestyle=':',
                linewidth=1.5, label=f'Budget (w={w:.2f})')

    # Optimum point
    if show_optimum:
        ax.plot(eq.consumption, eq.leisure, 'o', color='indianred', markersize=14,
                markeredgecolor='darkred', markeredgewidth=2,
                label=f'Optimum (C={eq.consumption:.1f}, R={eq.leisure:.1f})', zorder=10)

    ax.set_xlabel('Consumption (C)', fontsize=11)
    ax.set_ylabel('Leisure (R)', fontsize=11)
    ax.set_xlim(0, max(C_vals) * 1.1)
    ax.set_ylim(0, T * 1.05)
    ax.set_title(title or 'Robinson Crusoe Economy')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_ppf(
    rc: RobinsonCrusoe,
    ax: Optional[plt.Axes] = None,
    show_mrs_mrts: bool = True,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """
    Plot Production Possibility Frontier alone.
    
    Shows tradeoff between consumption and leisure.
    Slope = -MPL = opportunity cost of leisure in terms of consumption.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    T = rc.total_time
    eq = rc.find_equilibrium()
    
    # PPF
    R_vals = np.linspace(0.1, T - 0.1, 200)
    C_vals = np.array([rc.output(T - R) for R in R_vals])
    
    ax.plot(C_vals, R_vals, 'b-', linewidth=2.5)
    ax.fill_between(C_vals, 0, R_vals, alpha=0.15, color='blue', label='Feasible set')
    
    # Optimum
    ax.plot(eq.consumption, eq.leisure, 'ro', markersize=10,
            markeredgecolor='black', markeredgewidth=2)
    
    # Tangent line at optimum (slope = -1/MPL = -1/w)
    if show_mrs_mrts:
        slope = -1 / eq.wage  # dR/dC at optimum
        C_range = np.linspace(eq.consumption - 10, eq.consumption + 10, 50)
        R_tangent = eq.leisure + slope * (C_range - eq.consumption)
        mask = (R_tangent > 0) & (R_tangent < T) & (C_range > 0)
        ax.plot(C_range[mask], R_tangent[mask], 'g--', linewidth=1.5,
                label=f'Slope = -1/MPL = {slope:.3f}')
    
    ax.set_xlabel('Consumption')
    ax.set_ylabel('Leisure')
    ax.set_title('Production Possibility Frontier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, T)
    
    return ax
