"""
Plotting module for econoclasses.
"""

from .preferences import (
    PlotStyle,
    strategic_curve_levels,
    plot_indifference_curves,
    plot_slutsky_decomposition,
    plot_market_demand,
    plot_market_preferences
)

from .equilibrium import (
    plot_edgeworth_box,
    plot_supply_demand
)

from .production import (
    plot_isoquants,
    plot_firm_supply,
    plot_industry_supply,
)

from .general import (
    plot_robinson_crusoe,
    plot_ppf
)

from .welfare import (
    plot_upf,
    plot_social_indifference_curves,
)

__all__ = [
    # Preferences
    'PlotStyle',
    'strategic_curve_levels',
    'plot_indifference_curves',
    'plot_slutsky_decomposition',
    'plot_market_demand',
    'plot_market_preferences',
    # Equilibrium
    'plot_edgeworth_box',
    'plot_supply_demand',
    # Production
    'plot_isoquants',
    'plot_firm_supply',
    'plot_industry_supply',
    # General equilibrium
    'plot_robinson_crusoe',
    'plot_ppf',
    # Welfare
    'plot_upf',
    'plot_social_indifference_curves',
]
