"""
econoclasses v0.4.0
===================

Python toolkit for teaching undergraduate microeconomics.

Modules
-------
preferences : Utility functions and demand analysis
consumer : Individual agents and market aggregation  
production : Production functions, cost minimization, supply
equilibrium : Market clearing (exchange, partial, general)
welfare : Efficiency, social welfare functions, inequality
game_theory : Strategic interactions, Nash equilibrium, oligopoly
market_structures : Monopoly, perfect competition, price discrimination
externalities : Pigouvian taxes, Coase theorem, pollution
public_goods : Free rider problem, Lindahl equilibrium
plotting : Visualization tools
education : Step-by-step derivations
"""

__version__ = '0.4.0'

# Core symbols
from .core import X, Y, K, L, Px, Py, I, w, r

# Preferences
from .preferences import Utility

# Consumer
from .consumer import Consumer, Market

# Production
from .production import ProductionFunction, Firm, Industry

# Equilibrium
from .equilibrium import (
    ExchangeEconomy, PartialEquilibrium,
    RobinsonCrusoe, ProductionEconomy
)

# Game Theory
from .game_theory import (
    NormalFormGame, ExtensiveFormGame,
    find_pure_nash, find_mixed_nash_2player,
    cournot_duopoly, bertrand_duopoly, stackelberg_duopoly,
    prisoners_dilemma, battle_of_sexes
)

# Market Structures
from .market_structures import (
    LinearDemand, LinearCost, QuadraticCost,
    perfect_competition, monopoly,
    first_degree_discrimination, third_degree_discrimination,
    lerner_index, herfindahl_index,
    compare_market_structures, welfare_comparison_table
)

# Externalities
from .externalities import (
    ExternalityMarket, pigouvian_tax,
    coase_bargaining, cap_and_trade,
    pollution_example, education_spillover_example
)

# Public Goods
from .public_goods import (
    PublicGood, voluntary_provision, lindahl_equilibrium,
    contribution_game, compare_provision_mechanisms,
    national_defense_example, public_goods_game_experiment
)

__all__ = [
    # Symbols
    'X', 'Y', 'K', 'L', 'Px', 'Py', 'I', 'w', 'r',
    # Preferences
    'Utility',
    # Consumer
    'Consumer', 'Market',
    # Production
    'ProductionFunction', 'Firm', 'Industry',
    # Equilibrium
    'ExchangeEconomy', 'PartialEquilibrium',
    'RobinsonCrusoe', 'ProductionEconomy',
    # Game Theory
    'NormalFormGame', 'ExtensiveFormGame',
    'find_pure_nash', 'find_mixed_nash_2player',
    'cournot_duopoly', 'bertrand_duopoly', 'stackelberg_duopoly',
    'prisoners_dilemma', 'battle_of_sexes',
    # Market Structures
    'LinearDemand', 'LinearCost', 'QuadraticCost',
    'perfect_competition', 'monopoly',
    'first_degree_discrimination', 'third_degree_discrimination',
    'lerner_index', 'herfindahl_index',
    'compare_market_structures', 'welfare_comparison_table',
    # Externalities
    'ExternalityMarket', 'pigouvian_tax',
    'coase_bargaining', 'cap_and_trade',
    'pollution_example', 'education_spillover_example',
    # Public Goods
    'PublicGood', 'voluntary_provision', 'lindahl_equilibrium',
    'contribution_game', 'compare_provision_mechanisms',
    'national_defense_example', 'public_goods_game_experiment',
]
