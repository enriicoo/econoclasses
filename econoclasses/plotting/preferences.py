"""
Plotting for preferences and demand analysis.

Key conventions:
- Optimal point at center of frame (0.5, 0.5 of axes)
- Curves span 0.25-0.75 of the diagonal
- Middle curve passes exactly through the optimal point
- Background uses mid-range colormap (alpha 0.4), curves use extremes (0.6-1.0)

These conventions are shared with production plotting (isoquants).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Literal, Callable, Union
from dataclasses import dataclass

from ..preferences import Utility
from ..consumer import Market


@dataclass
class PlotStyle:
    """Configuration for plot appearance."""
    cmap: str = 'viridis'  # Match v0.1.0 default
    figsize: Tuple[int, int] = (8, 6)
    num_curves: int = 5
    show_colorbar: bool = True
    show_grid: bool = True
    linewidth: float = 4.0
    budget_color: str = 'red'
    optimal_color: str = 'red'
    budget_linestyle: str = '--'
    show_equation: bool = False
    curve_alpha: float = 1.0
    background_alpha: float = 0.4


def strategic_curve_levels(
    optimal_value: float,
    num_curves: int = 5,
    spread: float = 0.5,
) -> List[float]:
    """
    Generate curve levels ensuring the optimal value lies exactly on the middle curve.

    Parameters
    ----------
    optimal_value : float
        The value at the optimal point (utility, output, etc.).
    num_curves : int
        Total number of curves to display (should be odd for perfect centering).
    spread : float
        How far curves spread from optimal (0.5 means ±50% of optimal value).

    Returns
    -------
    List[float]
        Curve levels with the optimal value exactly at the middle position.
    """
    if num_curves < 1:
        return [optimal_value]

    # Middle index (0-indexed)
    mid_idx = (num_curves - 1) // 2

    # Generate levels such that optimal_value is exactly at mid_idx
    levels = []
    for i in range(num_curves):
        steps_from_mid = i - mid_idx
        step_size = spread * optimal_value / max(mid_idx, 1)
        levels.append(optimal_value + steps_from_mid * step_size)

    # Ensure all levels are positive
    min_level = min(levels)
    if min_level <= 0:
        shift = abs(min_level) + 0.1 * optimal_value
        levels = [lv + shift for lv in levels]

    return levels


def _compute_centered_frame_generic(
    x_star: float,
    y_star: float,
    x_fallback: float = 10.0,
    y_fallback: float = 10.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute plot frame with optimal point at CENTER (0.5, 0.5 of axes).

    This is the generic version used by both preferences and production.
    """
    if x_star > 0 and y_star > 0:
        x_max = x_star * 2.0
        y_max = y_star * 2.0
    elif x_star == 0 and y_star > 0:
        x_max = y_star * 0.5
        y_max = y_star * 2.0
    elif x_star > 0 and y_star == 0:
        x_max = x_star * 2.0
        y_max = x_star * 0.5
    else:
        x_max = x_fallback
        y_max = y_fallback

    x_max = max(x_max, 1.0)
    y_max = max(y_max, 1.0)

    return (0.01, x_max), (0.01, y_max)


def _plot_contour_core(
    ax: plt.Axes,
    value_func: Callable,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    levels: List[float],
    style: PlotStyle,
    colorbar_label: str = 'Value',
    label_fmt: str = '%.2f',
) -> plt.Axes:
    """
    Core contour plotting logic shared by indifference curves and isoquants.

    This ensures identical visual behavior for both plot types.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on.
    value_func : Callable
        Function(x, y) -> value (utility or output).
    x_range, y_range : Tuple[float, float]
        Plot ranges.
    levels : List[float]
        Contour levels to plot.
    style : PlotStyle
        Styling configuration.
    colorbar_label : str
        Label for the colorbar.
    label_fmt : str
        Format string for contour labels.

    Returns
    -------
    plt.Axes
        The matplotlib axes with contours plotted.
    """
    # Create grid (300 points for smooth curves, matching v0.1.0)
    x = np.linspace(x_range[0], x_range[1], 300)
    y = np.linspace(y_range[0], y_range[1], 300)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Evaluate function on grid
    Z_grid = value_func(X_grid, Y_grid)
    Z_grid = np.where(np.isfinite(Z_grid), Z_grid, np.nan)

    cmap_obj = plt.get_cmap(style.cmap)

    # Background: value gradient using colormap with transparency
    if style.show_colorbar:
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        im = ax.imshow(
            Z_grid, extent=extent, origin='lower',
            cmap=style.cmap, aspect='auto', alpha=style.background_alpha
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_title(colorbar_label, fontsize=10)

    # Curves: use extremes of colormap (0.6-1.0 range for visibility)
    # This matches v0.1.0: colors = cmap_obj(np.linspace(0.6, 1.0, len(levels)))
    colors = [cmap_obj(0.6 + 0.4 * i / max(len(levels) - 1, 1))
              for i in range(len(levels))]

    cs = ax.contour(
        X_grid, Y_grid, Z_grid,
        levels=levels,
        colors=colors,
        linewidths=style.linewidth,
        alpha=style.curve_alpha
    )

    return ax, cs, cmap_obj




def _compute_centered_frame(
    x_star: float,
    y_star: float,
    utility: Optional[Utility] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute plot frame with optimal point at CENTER (0.5, 0.5 of axes).

    The frame ensures:
    - Optimal point is at center of plot
    - Curves span approximately 0.25-0.75 of the diagonal
    """
    # The optimal point should be at 50% of each axis
    # So x_max = 2 * x_star, y_max = 2 * y_star

    if x_star > 0 and y_star > 0:
        # Normal interior solution: center on optimal
        x_max = x_star * 2.0
        y_max = y_star * 2.0
    elif x_star == 0 and y_star > 0:
        # Corner solution on Y axis
        # Put optimal at center-left, but need some X range
        if utility is not None:
            x_max = utility.income / utility.price_x * 0.4
        else:
            x_max = y_star * 0.5
        y_max = y_star * 2.0
    elif x_star > 0 and y_star == 0:
        # Corner solution on X axis
        x_max = x_star * 2.0
        if utility is not None:
            y_max = utility.income / utility.price_y * 0.4
        else:
            y_max = x_star * 0.5
    else:
        # Both zero - fallback
        if utility is not None:
            x_max = utility.income / utility.price_x
            y_max = utility.income / utility.price_y
        else:
            x_max = 10.0
            y_max = 10.0

    # Ensure minimum visibility
    x_max = max(x_max, 1.0)
    y_max = max(y_max, 1.0)

    return (0.01, x_max), (0.01, y_max)


def plot_indifference_curves(
    utility: Utility,
    ax: Optional[plt.Axes] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    style: Optional[PlotStyle] = None,
    show_budget: bool = False,
    show_optimal: bool = False,
    show_equation: bool = False,
    title: Optional[str] = None,
    utility_levels: Optional[List[float]] = None,
    label_curves: Union[bool, Literal['interpolate', 'center']] = False,
    label_position: Literal['first', 'mid', 'last'] = 'mid',
    label_direction: Literal['dynamic', 'horizontal', 'vertical', 'diagonal'] = 'dynamic',
) -> plt.Axes:
    """
    Plot indifference curves for a utility function.

    Key features:
    - Optimal point centered at (0.5, 0.5) of frame
    - Middle curve passes through the optimal bundle
    - Background gradient, curves at colormap extremes (0.6-1.0)

    Parameters
    ----------
    utility : Utility
        The utility function to plot.
    ax : matplotlib Axes, optional
        Axes to draw on. Creates a new figure if None.
    x_range, y_range : (float, float), optional
        Plot limits for each axis. Auto-computed from the optimal bundle if None.
    style : PlotStyle, optional
        Visual configuration (colormap, linewidth, number of curves, etc.).
    show_budget : bool
        Draw the budget constraint line.
    show_optimal : bool
        Mark the optimal consumption bundle and display an info box.
    show_equation : bool
        Show the utility function formula as the plot title (LaTeX).
    title : str, optional
        Custom plot title. Overrides show_equation.
    utility_levels : list of float, optional
        Explicit utility levels to draw. Auto-selected if None.
    label_curves : False | True | 'interpolate' | 'center'
        False          → no labels
        True           → label every curve
        'interpolate'  → label every other curve, middle always included
        'center'       → label the middle curve only
    label_position : 'first' | 'mid' | 'last'
        Where along each curve path to place the label.
        'first'  → ~25% along the path (near one end)
        'mid'    → ~50% along the path (centre)
        'last'   → ~75% along the path (near the other end)
    label_direction : 'dynamic' | 'horizontal' | 'vertical' | 'diagonal'
        Rotation of the label text.
        'dynamic'    → follows the curve tangent (matplotlib default)
        'horizontal' → always flat (0°)
        'vertical'   → always upright (90°)
        'diagonal'   → fixed 45° (bottom-left to top-right)
    """
    style = style or PlotStyle()

    x_star, y_star = utility.marshallian_demand
    u_star = utility.utility_at(x_star, y_star)

    # Compute centered frame if not provided
    if x_range is None or y_range is None:
        auto_x, auto_y = _compute_centered_frame(x_star, y_star, utility)
        x_range = x_range or auto_x
        y_range = y_range or auto_y

    if ax is None:
        fig, ax = plt.subplots(figsize=style.figsize)

    # Strategic curve levels: optimal utility on middle curve
    if utility_levels is None:
        utility_levels = strategic_curve_levels(u_star, style.num_curves, spread=0.5)

    # Use shared core plotting logic
    ax, cs, cmap_obj = _plot_contour_core(
        ax=ax,
        value_func=utility._utility_func,
        x_range=x_range,
        y_range=y_range,
        levels=utility_levels,
        style=style,
        colorbar_label='U',
    )

    # Label curves if requested
    if label_curves is not False:
        n = len(cs.allsegs)
        mid = n // 2
        if label_curves is True:
            indices = list(range(n))
        elif label_curves == 'center':
            indices = [mid]
        else:  # 'interpolate': every other, middle always included
            indices = list(range(mid % 2, n, 2))

        label_points = []
        for i, segs in enumerate(cs.allsegs):
            if i not in indices:
                continue
            for seg in segs:
                if len(seg) > 0:
                    if label_position == 'first':
                        idx = len(seg) // 4
                    elif label_position == 'last':
                        idx = (len(seg) // 4) * 3
                    else:
                        idx = len(seg) // 2
                    idx = min(idx, len(seg) - 1)
                    label_points.append(seg[idx])
                    break

        if label_points:
            labels = ax.clabel(cs, inline=True, fmt="%.2f", fontsize=16, manual=label_points)
            for lbl in labels:
                lbl.set_fontweight('bold')
                if label_direction == 'horizontal':
                    lbl.set_rotation(0)
                elif label_direction == 'vertical':
                    lbl.set_rotation(90)
                elif label_direction == 'diagonal':
                    lbl.set_rotation(45)

    # Legend: text entries first so they appear at the top
    if show_optimal:
        ax.plot([], [], ' ', label=f'$P_x={utility.price_x:.2f}$   $P_y={utility.price_y:.2f}$')
        ax.plot([], [], ' ', label=f'$X^*={x_star:.2f}$   $Y^*={y_star:.2f}$')

    # Budget constraint
    if show_budget:
        x_intercept = utility.income / utility.price_x
        budget_x = np.linspace(0, min(x_intercept, x_range[1]), 100)
        budget_y = (utility.income - utility.price_x * budget_x) / utility.price_y
        mask = (budget_y >= 0) & (budget_y <= y_range[1])

        mid_color = cmap_obj(0.5)
        inv_color = (1 - mid_color[0], 1 - mid_color[1], 1 - mid_color[2])

        budget_label = (f'Budget   $U^*={u_star:.2f}$' if show_optimal else 'Budget')
        ax.plot(
            budget_x[mask], budget_y[mask],
            color=inv_color, linestyle=style.budget_linestyle,
            linewidth=2.5, label=budget_label
        )

    # Optimal point
    if show_optimal:
        mid_color = cmap_obj(0.5)
        inv_color = (1 - mid_color[0], 1 - mid_color[1], 1 - mid_color[2])

        ax.plot(
            x_star, y_star, 'o',
            color=inv_color,
            markersize=12, markeredgecolor='black', markeredgewidth=1.5,
            label=f'Optimal ({x_star:.2f}, {y_star:.2f})', zorder=10
        )

    ax.set_xlabel('Quantity of Good X')
    ax.set_ylabel('Quantity of Good Y')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Title with LaTeX equation support
    if title:
        ax.set_title(title)
    elif show_equation and hasattr(utility, 'get_equation_latex'):
        eq_latex = utility.get_equation_latex()
        ax.set_title(f'${eq_latex}$')
    else:
        ax.set_title(f'Indifference Curves: {utility.form_name}')

    if style.show_grid:
        ax.grid(True, alpha=0.3)

    if show_budget or show_optimal:
        ax.legend(loc='upper right')

    return ax


def plot_slutsky_decomposition(
    utility: Utility,
    new_price_x: float,
    ax: Optional[plt.Axes] = None,
    style: Optional[PlotStyle] = None,
    num_curves: int = 4,
) -> plt.Axes:
    """
    Visualize Slutsky decomposition with black/grey tones and red accent.

    Shows 4 curves: 2 around original utility, 2 around new utility.
    """
    style = style or PlotStyle()

    decomp = utility.slutsky_decomposition(new_price_x=new_price_x, good='X')
    A, B, C = decomp['original_bundle'], decomp['compensated_bundle'], decomp['new_bundle']
    u_original = utility.utility_at(*A)
    u_new = utility.utility_at(*C)

    # Frame centered on the midpoint of A and C
    mid_x = (A[0] + C[0]) / 2
    mid_y = (A[1] + C[1]) / 2

    # Make frame so midpoint is at center
    x_max = mid_x * 2.0
    y_max = mid_y * 2.0

    # Ensure budget intercepts are visible
    x_int_old = utility.income / utility.price_x
    y_int_old = utility.income / utility.price_y
    x_int_new = utility.income / new_price_x

    x_max = max(x_max, x_int_old * 1.05, x_int_new * 1.05)
    y_max = max(y_max, y_int_old * 1.05)

    if ax is None:
        fig, ax = plt.subplots(figsize=style.figsize)

    x = np.linspace(0.01, x_max, 300)
    y = np.linspace(0.01, y_max, 300)
    X_grid, Y_grid = np.meshgrid(x, y)
    U_grid = utility._utility_func(X_grid, Y_grid)

    # 2 curves: original utility (black) and new utility (red)
    ax.contour(X_grid, Y_grid, U_grid, levels=[u_original],
               colors=['black'], linewidths=2.5)
    ax.contour(X_grid, Y_grid, U_grid, levels=[u_new],
               colors=['indianred'], linewidths=2.5)

    # Budget lines in grey/black
    ax.plot([0, x_int_old], [y_int_old, 0], color='black', linestyle='--',
            linewidth=2, label='Original Budget')
    ax.plot([0, x_int_new], [y_int_old, 0], color='indianred', linestyle='--',
            linewidth=2, label='New Budget')

    # Compensated budget in grey
    comp_income = new_price_x * B[0] + utility.price_y * B[1]
    x_int_comp = comp_income / new_price_x
    y_int_comp = comp_income / utility.price_y
    ax.plot([0, x_int_comp], [y_int_comp, 0], color='dimgrey', linestyle=':',
            linewidth=2, label='Compensated')

    # Points
    ax.plot(*A, 'ko', markersize=14, markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.plot(*B, 'o', color='dimgrey', markersize=14, markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.plot(*C, 'o', color='indianred', markersize=14, markeredgecolor='white', markeredgewidth=2, zorder=10)

    ax.annotate('A (Original)', A, xytext=(10, 10), textcoords='offset points', fontsize=10)
    ax.annotate('B (Compensated)', B, xytext=(10, -15), textcoords='offset points', fontsize=10)
    ax.annotate('C (New)', C, xytext=(10, 10), textcoords='offset points', fontsize=10)

    ax.set_xlabel('Quantity of Good X')
    ax.set_ylabel('Quantity of Good Y')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title(f'Slutsky Decomposition: $P_x$: {utility.price_x} $\\rightarrow$ {new_price_x}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Decomposition info box
    textstr = '\n'.join([
        f'Substitution: {decomp["substitution_effect"]:+.2f}',
        f'Income: {decomp["income_effect"]:+.2f}',
        f'Total: {decomp["total_effect"]:+.2f}'
    ])
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    )

    return ax


def plot_market_demand(
    market: Market,
    good: str = 'X',
    price_range: Tuple[float, float] = (0.5, 10),
    other_price: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show_individual: bool = True,
    cmap: str = 'viridis',
    n_points: int = 200,
) -> plt.Axes:
    """Plot individual and aggregate demand curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Use more points to capture kinks in demand (e.g., perfect substitutes)
    schedule = market.demand_schedule(good, price_range, other_price, n_points=n_points)
    prices = schedule['prices']

    cmap_obj = plt.get_cmap(cmap)

    if show_individual:
        n_consumers = len(market.consumers)
        for i, consumer in enumerate(market.consumers):
            color = cmap_obj(0.3 + 0.5 * i / max(n_consumers - 1, 1))
            ax.plot(schedule[consumer.name], prices, alpha=0.7, linewidth=1.5,
                    color=color, label=consumer.name)

    ax.plot(schedule['aggregate'], prices, 'k-', linewidth=2.5, label='Market')

    ax.set_xlabel(f'Quantity of {good}')
    ax.set_ylabel(f'Price of {good}')
    ax.set_title(f'Market Demand for {good}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_market_preferences(
    market: Market,
    prices: Tuple[float, float] = (2.0, 1.0),
    ax: Optional[plt.Axes] = None,
    show_individual: bool = True,
    cmap: str = 'viridis',
    individual_colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    n_price_points: int = 100,
) -> plt.Axes:
    """
    Plot market preferences: individual ICs aggregating into market IC.

    Shows how individual consumer indifference curves combine to form
    the aggregate market preference in goods space (X-Y).

    The market IC is computed by tracing out aggregate demand at various
    price ratios (holding market expenditure constant). This gives the
    true aggregate demand behavior rather than assuming a functional form.

    Parameters
    ----------
    market : Market
        Market with multiple consumers.
    prices : Tuple[float, float]
        Reference prices (Px, Py) at which to compute optimal bundles.
    show_individual : bool
        If True, show individual consumer ICs and optimal points.
    individual_colors : list, optional
        Colors for individual consumers. If None, uses default palette.
    n_price_points : int
        Number of price ratios to sample for tracing the market IC.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    px, py = prices

    # Compute individual optimal bundles at reference prices
    optima = []
    for consumer in market.consumers:
        x, y = consumer.utility.demand_at_prices(px, py, consumer.income)
        optima.append((x, y))

    # Aggregate bundle at reference prices
    agg_x = sum(opt[0] for opt in optima)
    agg_y = sum(opt[1] for opt in optima)

    # Trace out the market IC by varying price ratio
    # At each price ratio, compute what each consumer demands given their income,
    # then sum to get the aggregate bundle
    market_ic_x = []
    market_ic_y = []

    # Vary price ratio from very low px/py to very high px/py
    # Use log scale for better coverage of price ratios
    price_ratios = np.logspace(-1.5, 1.5, n_price_points)  # px/py from ~0.03 to ~32

    for ratio in price_ratios:
        # Set prices such that px/py = ratio, keeping py = 1 as numeraire
        test_py = 1.0
        test_px = ratio * test_py

        # Compute aggregate demand at these prices
        total_x = 0.0
        total_y = 0.0
        for consumer in market.consumers:
            cx, cy = consumer.utility.demand_at_prices(test_px, test_py, consumer.income)
            total_x += cx
            total_y += cy

        # Include points where both goods are positive and within reasonable bounds
        # (don't include extreme corner solutions that would distort the plot)
        if total_x > 0 and total_y > 0:
            # Limit to 5x the aggregate bundle to keep plot readable
            if total_x <= agg_x * 5 and total_y <= agg_y * 5:
                market_ic_x.append(total_x)
                market_ic_y.append(total_y)

    market_ic_x = np.array(market_ic_x)
    market_ic_y = np.array(market_ic_y)

    # Center frame on aggregate bundle (2x gives optimal point at center)
    x_max = agg_x * 2.0
    y_max = agg_y * 2.0
    x_vals = np.linspace(0.1, x_max, 200)
    y_vals = np.linspace(0.1, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Plot individual ICs (dashed, behind)
    if show_individual:
        if individual_colors is None:
            individual_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        for i, (consumer, opt) in enumerate(zip(market.consumers, optima)):
            color = individual_colors[i % len(individual_colors)]
            U_grid = consumer.utility._utility_func(X_grid, Y_grid)
            u_star = consumer.utility.utility_at(*opt)

            # Individual IC
            ax.contour(X_grid, Y_grid, U_grid, levels=[u_star],
                       colors=[color], linewidths=2, linestyles='--', alpha=0.7)

            # Individual optimal point
            ax.plot(*opt, 'o', color=color, markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, label=f'{consumer.name} ({opt[0]:.1f}, {opt[1]:.1f})')

            # Line to aggregate
            ax.plot([opt[0], agg_x], [opt[1], agg_y], ':', color=color, alpha=0.4, linewidth=1)

    # Market IC (solid, prominent) - traced from actual aggregate demands
    cmap_obj = plt.get_cmap(cmap)
    if len(market_ic_x) > 1:
        # Sort by x for clean line plotting
        sort_idx = np.argsort(market_ic_x)
        ax.plot(market_ic_x[sort_idx], market_ic_y[sort_idx],
                color=cmap_obj(0.8), linewidth=3, alpha=1.0,
                label='Market IC (aggregate demand)')

    # Aggregate point (black dot)
    ax.plot(agg_x, agg_y, 'ko', markersize=14, markeredgecolor='white',
            markeredgewidth=2, label=f'Market ({agg_x:.1f}, {agg_y:.1f})', zorder=10)

    ax.set_xlabel('Good X')
    ax.set_ylabel('Good Y')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title(title or f'Aggregate Preferences (Px={px}, Py={py})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax
