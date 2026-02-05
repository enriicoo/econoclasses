"""
Game Theory Module
==================

Strategic interaction analysis for microeconomics.

Classes
-------
NormalFormGame : Normal (strategic) form game representation
ExtensiveFormGame : Extensive (tree) form game representation  
Player, Strategy : Game components

Solvers
-------
find_pure_nash : Find pure strategy Nash equilibria
find_mixed_nash_2player : Find mixed strategy equilibria (2-player)
find_all_nash : Find all equilibria
find_dominant_strategy_equilibrium : Find dominant strategy equilibrium
iterated_elimination : IESDS/IEWDS
maxmin_strategy, solve_zero_sum : Security strategies and zero-sum solutions

Classic Games
-------------
prisoners_dilemma, battle_of_sexes, matching_pennies, stag_hunt, 
chicken, rock_paper_scissors, pure_coordination

Economic Applications  
---------------------
cournot_duopoly : Quantity competition
bertrand_duopoly : Price competition
stackelberg_duopoly : Sequential quantity competition
cournot_n_firms : N-firm Cournot
entry_game, ultimatum_game : Extensive form examples

Example
-------
>>> from econoclasses.game_theory import prisoners_dilemma, find_pure_nash
>>> 
>>> game = prisoners_dilemma()
>>> print(game.payoff_table())
>>> 
>>> equilibria = find_pure_nash(game)
>>> print(equilibria)
[PureNashEq(('Defect', 'Defect'), payoffs=(-2, -2))]
"""

# Core game representations
from .games import (
    Strategy,
    Player,
    StrategyProfile,
    Outcome,
    NormalFormGame,
    GameNode,
    ExtensiveFormGame,
)

# Equilibrium solvers
from .solvers import (
    PureNashEquilibrium,
    MixedNashEquilibrium,
    DominantStrategyEquilibrium,
    find_pure_nash,
    find_mixed_nash_2player,
    find_all_nash,
    find_dominant_strategy_equilibrium,
    iterated_elimination,
    maxmin_strategy,
    minmax_value,
    solve_zero_sum,
)

# Classic games
from .classics import (
    # 2x2 games
    prisoners_dilemma,
    battle_of_sexes,
    matching_pennies,
    stag_hunt,
    chicken,
    rock_paper_scissors,
    pure_coordination,
    
    # Oligopoly models
    CournotResult,
    BertrandResult,
    StackelbergResult,
    cournot_duopoly,
    cournot_best_response,
    cournot_n_firms,
    bertrand_duopoly,
    stackelberg_duopoly,
    
    # Extensive form examples
    entry_game,
    ultimatum_game,
    centipede_game,
    
    # Analysis
    compare_oligopoly_models,
)

__all__ = [
    # Core
    'Strategy',
    'Player', 
    'StrategyProfile',
    'Outcome',
    'NormalFormGame',
    'GameNode',
    'ExtensiveFormGame',
    
    # Solvers
    'PureNashEquilibrium',
    'MixedNashEquilibrium', 
    'DominantStrategyEquilibrium',
    'find_pure_nash',
    'find_mixed_nash_2player',
    'find_all_nash',
    'find_dominant_strategy_equilibrium',
    'iterated_elimination',
    'maxmin_strategy',
    'minmax_value',
    'solve_zero_sum',
    
    # Classic games
    'prisoners_dilemma',
    'battle_of_sexes',
    'matching_pennies',
    'stag_hunt',
    'chicken',
    'rock_paper_scissors',
    'pure_coordination',
    
    # Oligopoly
    'CournotResult',
    'BertrandResult',
    'StackelbergResult',
    'cournot_duopoly',
    'cournot_best_response',
    'cournot_n_firms',
    'bertrand_duopoly',
    'stackelberg_duopoly',
    
    # Extensive form
    'entry_game',
    'ultimatum_game',
    'centipede_game',
    
    # Analysis
    'compare_oligopoly_models',
]
