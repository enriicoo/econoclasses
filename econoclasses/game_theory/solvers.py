"""
Game theory solvers: Nash equilibrium, dominant strategy equilibrium.

Implements algorithms for finding equilibria in normal form games.
"""

import numpy as np
from scipy.optimize import linprog, minimize
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from itertools import product

from .games import NormalFormGame, Strategy


@dataclass
class PureNashEquilibrium:
    """A pure strategy Nash equilibrium."""
    strategies: Tuple[int, ...]  # Strategy indices
    strategy_names: Tuple[str, ...]
    payoffs: Tuple[float, ...]
    is_strict: bool  # True if strictly best responses
    
    def __repr__(self):
        return f"PureNashEq({self.strategy_names}, payoffs={self.payoffs})"


@dataclass 
class MixedNashEquilibrium:
    """A mixed strategy Nash equilibrium."""
    probabilities: List[np.ndarray]  # One probability vector per player
    expected_payoffs: Tuple[float, ...]
    support: List[List[int]]  # Indices of strategies in support
    
    def __repr__(self):
        probs_str = [f"[{', '.join(f'{p:.3f}' for p in probs)}]" 
                    for probs in self.probabilities]
        return f"MixedNashEq({probs_str}, payoffs={self.expected_payoffs})"


@dataclass
class DominantStrategyEquilibrium:
    """Equilibrium where all players play dominant strategies."""
    strategies: Tuple[int, ...]
    strategy_names: Tuple[str, ...]
    payoffs: Tuple[float, ...]
    dominance_type: str  # 'strict' or 'weak'
    
    def __repr__(self):
        return f"DominantStratEq({self.strategy_names}, type={self.dominance_type})"


# =============================================================================
# PURE STRATEGY NASH EQUILIBRIUM
# =============================================================================

def find_pure_nash(game: NormalFormGame, 
                   include_weak: bool = True) -> List[PureNashEquilibrium]:
    """
    Find all pure strategy Nash equilibria.
    
    Uses best response checking for all strategy profiles.
    
    Parameters
    ----------
    game : NormalFormGame
        The game to solve
    include_weak : bool
        If True, include equilibria where players are indifferent
    
    Returns
    -------
    list of PureNashEquilibrium
    
    Example
    -------
    >>> game = prisoners_dilemma()
    >>> eqs = find_pure_nash(game)
    >>> print(eqs[0])
    PureNashEq(('D', 'D'), payoffs=(-2, -2))
    """
    equilibria = []
    
    # Check each strategy profile
    for profile_indices in product(*[range(p.n_strategies) for p in game.players]):
        is_nash = True
        is_strict = True
        
        for player in range(game.n_players):
            # Get strategies of other players
            other_strats = {i: profile_indices[i] for i in range(game.n_players) if i != player}
            
            # Find best responses
            best_responses = game.best_response(player, other_strats)
            
            # Check if current strategy is a best response
            current_strat = profile_indices[player]
            if current_strat not in best_responses:
                is_nash = False
                break
            
            # Check if it's strictly best (unique best response)
            if len(best_responses) > 1:
                is_strict = False
        
        if is_nash:
            if is_strict or include_weak:
                strategy_names = tuple(
                    game.players[i].strategies[profile_indices[i]].name
                    for i in range(game.n_players)
                )
                payoffs = game.all_payoffs(profile_indices)
                equilibria.append(PureNashEquilibrium(
                    strategies=profile_indices,
                    strategy_names=strategy_names,
                    payoffs=payoffs,
                    is_strict=is_strict
                ))
    
    return equilibria


# =============================================================================
# MIXED STRATEGY NASH EQUILIBRIUM (2-Player)
# =============================================================================

def find_mixed_nash_2player(game: NormalFormGame, 
                            tol: float = 1e-8) -> List[MixedNashEquilibrium]:
    """
    Find mixed strategy Nash equilibria for 2-player games.
    
    Uses support enumeration algorithm: for each pair of supports,
    solve the indifference conditions.
    
    Parameters
    ----------
    game : NormalFormGame
        A 2-player game
    tol : float
        Tolerance for numerical comparisons
    
    Returns
    -------
    list of MixedNashEquilibrium
    """
    if game.n_players != 2:
        raise ValueError("This method only works for 2-player games")
    
    A = game.payoff_arrays[0]  # Row player payoffs
    B = game.payoff_arrays[1]  # Column player payoffs
    m, n = A.shape
    
    equilibria = []
    
    # Enumerate all possible support pairs
    from itertools import combinations
    
    for k1 in range(1, m + 1):
        for k2 in range(1, n + 1):
            for support1 in combinations(range(m), k1):
                for support2 in combinations(range(n), k2):
                    eq = _solve_support_enumeration(A, B, support1, support2, tol)
                    if eq is not None:
                        equilibria.append(eq)
    
    # Remove duplicates
    unique_equilibria = []
    for eq in equilibria:
        is_duplicate = False
        for existing in unique_equilibria:
            if (np.allclose(eq.probabilities[0], existing.probabilities[0], atol=tol) and
                np.allclose(eq.probabilities[1], existing.probabilities[1], atol=tol)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_equilibria.append(eq)
    
    return unique_equilibria


def _solve_support_enumeration(A: np.ndarray, B: np.ndarray,
                                support1: Tuple[int, ...], 
                                support2: Tuple[int, ...],
                                tol: float = 1e-8) -> Optional[MixedNashEquilibrium]:
    """
    Try to find equilibrium with given supports.
    
    Solve: 
    - q makes player 1 indifferent over support1
    - p makes player 2 indifferent over support2
    """
    m, n = A.shape
    k1, k2 = len(support1), len(support2)
    
    # For player 1: find q such that all strategies in support1 give same expected payoff
    # A[i, :] @ q = v for all i in support1
    # sum(q) = 1, q >= 0
    
    # Set up linear system for player 2's mixing (makes player 1 indifferent)
    # We need: A[support1[0], support2] @ q = A[support1[i], support2] @ q for all i
    
    if k2 == 1:
        # Pure strategy for player 2
        q = np.zeros(n)
        q[support2[0]] = 1.0
    else:
        # Build system of indifference equations
        A_sub = A[np.ix_(list(support1), list(support2))]
        
        # Indifference: (A_sub[0] - A_sub[i]) @ q = 0 for i > 0
        # Plus: sum(q) = 1
        eq_matrix = np.vstack([
            A_sub[0] - A_sub[i] for i in range(1, k1)
        ] + [np.ones(k2)])
        eq_rhs = np.array([0] * (k1 - 1) + [1])
        
        # Solve if square, least squares otherwise
        try:
            if eq_matrix.shape[0] == eq_matrix.shape[1]:
                q_support = np.linalg.solve(eq_matrix, eq_rhs)
            else:
                q_support, _, _, _ = np.linalg.lstsq(eq_matrix, eq_rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        # Check validity: all probabilities in [0, 1]
        if np.any(q_support < -tol) or np.any(q_support > 1 + tol):
            return None
        
        q_support = np.clip(q_support, 0, 1)
        q_support = q_support / q_support.sum()  # Renormalize
        
        q = np.zeros(n)
        for i, idx in enumerate(support2):
            q[idx] = q_support[i]
    
    # Similarly for player 1's mixing (makes player 2 indifferent)
    if k1 == 1:
        p = np.zeros(m)
        p[support1[0]] = 1.0
    else:
        B_sub = B[np.ix_(list(support1), list(support2))]
        
        # Player 2's payoff from column j: p @ B_sub[:, j]
        # Indifference: p @ B_sub[:, 0] = p @ B_sub[:, j] for j > 0
        eq_matrix = np.vstack([
            (B_sub[:, 0] - B_sub[:, j]).T for j in range(1, k2)
        ] + [np.ones((1, k1))])
        eq_rhs = np.array([0] * (k2 - 1) + [1])
        
        try:
            if eq_matrix.shape[0] == eq_matrix.shape[1]:
                p_support = np.linalg.solve(eq_matrix, eq_rhs)
            else:
                p_support, _, _, _ = np.linalg.lstsq(eq_matrix, eq_rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        if np.any(p_support < -tol) or np.any(p_support > 1 + tol):
            return None
        
        p_support = np.clip(p_support, 0, 1)
        p_support = p_support / p_support.sum()
        
        p = np.zeros(m)
        for i, idx in enumerate(support1):
            p[idx] = p_support[i]
    
    # Verify this is actually a Nash equilibrium
    # Check player 1: strategies in support should be equally good, 
    #                 strategies outside should be no better
    exp_payoffs_1 = A @ q
    support_payoff = exp_payoffs_1[support1[0]]
    
    for i in range(m):
        if i in support1:
            if not np.isclose(exp_payoffs_1[i], support_payoff, atol=tol):
                return None
        else:
            if exp_payoffs_1[i] > support_payoff + tol:
                return None
    
    # Check player 2
    exp_payoffs_2 = B.T @ p
    support_payoff_2 = exp_payoffs_2[support2[0]]
    
    for j in range(n):
        if j in support2:
            if not np.isclose(exp_payoffs_2[j], support_payoff_2, atol=tol):
                return None
        else:
            if exp_payoffs_2[j] > support_payoff_2 + tol:
                return None
    
    # Compute expected payoffs
    exp_payoff_1 = p @ A @ q
    exp_payoff_2 = p @ B @ q
    
    return MixedNashEquilibrium(
        probabilities=[p, q],
        expected_payoffs=(exp_payoff_1, exp_payoff_2),
        support=[list(support1), list(support2)]
    )


def find_all_nash(game: NormalFormGame) -> Dict[str, List]:
    """
    Find all Nash equilibria (pure and mixed) for a 2-player game.
    
    Returns
    -------
    dict with keys 'pure' and 'mixed'
    """
    pure = find_pure_nash(game)
    mixed = find_mixed_nash_2player(game) if game.n_players == 2 else []
    
    return {
        'pure': pure,
        'mixed': mixed
    }


# =============================================================================
# DOMINANT STRATEGY EQUILIBRIUM
# =============================================================================

def find_dominant_strategy_equilibrium(game: NormalFormGame,
                                       strict: bool = True) -> Optional[DominantStrategyEquilibrium]:
    """
    Find dominant strategy equilibrium if one exists.
    
    Parameters
    ----------
    game : NormalFormGame
        The game to solve
    strict : bool
        If True, require strictly dominant strategies
    
    Returns
    -------
    DominantStrategyEquilibrium or None
    """
    dominant_strategies = []
    
    for player in range(game.n_players):
        dom_strat = game.dominant_strategy(player)
        if dom_strat is None:
            return None
        dominant_strategies.append(dom_strat)
    
    strategy_names = tuple(
        game.players[i].strategies[dominant_strategies[i]].name
        for i in range(game.n_players)
    )
    payoffs = game.all_payoffs(tuple(dominant_strategies))
    
    return DominantStrategyEquilibrium(
        strategies=tuple(dominant_strategies),
        strategy_names=strategy_names,
        payoffs=payoffs,
        dominance_type='strict' if strict else 'weak'
    )


# =============================================================================
# ITERATED ELIMINATION OF DOMINATED STRATEGIES
# =============================================================================

def iterated_elimination(game: NormalFormGame, 
                         strict: bool = True,
                         verbose: bool = False) -> Tuple[NormalFormGame, List[str]]:
    """
    Iteratively eliminate (strictly/weakly) dominated strategies.
    
    Parameters
    ----------
    game : NormalFormGame
        The game to reduce
    strict : bool
        If True, only eliminate strictly dominated strategies
    verbose : bool
        If True, print elimination steps
    
    Returns
    -------
    tuple of (reduced_game, elimination_log)
    """
    # Work with copies of payoff arrays and track remaining strategies
    remaining = [list(range(p.n_strategies)) for p in game.players]
    current_payoffs = [arr.copy() for arr in game.payoff_arrays]
    log = []
    
    changed = True
    round_num = 0
    
    while changed:
        changed = False
        round_num += 1
        
        for player in range(game.n_players):
            # Check each remaining strategy
            to_remove = []
            
            for i, strat in enumerate(remaining[player]):
                # Check if dominated
                for j, other in enumerate(remaining[player]):
                    if i == j:
                        continue
                    
                    # Check if 'other' dominates 'strat'
                    dominates = True
                    strictly_better = False
                    
                    # Build indices for other players
                    other_ranges = [remaining[p] if p != player else [0] 
                                  for p in range(game.n_players)]
                    
                    for idx_combo in product(*other_ranges):
                        # Get actual indices in current arrays
                        full_idx_strat = list(idx_combo)
                        full_idx_strat[player] = i
                        full_idx_other = list(idx_combo)
                        full_idx_other[player] = j
                        
                        payoff_strat = current_payoffs[player][tuple(full_idx_strat)]
                        payoff_other = current_payoffs[player][tuple(full_idx_other)]
                        
                        if strict:
                            if payoff_other <= payoff_strat:
                                dominates = False
                                break
                        else:
                            if payoff_other < payoff_strat:
                                dominates = False
                                break
                            if payoff_other > payoff_strat:
                                strictly_better = True
                    
                    if dominates and (strict or strictly_better):
                        to_remove.append(strat)
                        strat_name = game.players[player].strategies[strat].name
                        other_name = game.players[player].strategies[other].name
                        msg = f"Round {round_num}: Player {player} strategy '{strat_name}' dominated by '{other_name}'"
                        log.append(msg)
                        if verbose:
                            print(msg)
                        break
            
            # Remove dominated strategies
            for strat in set(to_remove):
                if strat in remaining[player]:
                    remaining[player].remove(strat)
                    changed = True
    
    # Build reduced game
    if all(len(r) == game.players[i].n_strategies for i, r in enumerate(remaining)):
        return game, log
    
    # Create new game with remaining strategies
    new_strat_names = [
        [game.players[i].strategies[j].name for j in remaining[i]]
        for i in range(game.n_players)
    ]
    
    # Build new payoff dict
    new_payoffs = {}
    for combo_idx in product(*[range(len(r)) for r in remaining]):
        original_combo = tuple(remaining[i][combo_idx[i]] for i in range(game.n_players))
        strat_names = tuple(new_strat_names[i][combo_idx[i]] for i in range(game.n_players))
        new_payoffs[strat_names] = game.all_payoffs(original_combo)
    
    reduced_game = NormalFormGame.from_payoff_dict(
        [p.name for p in game.players],
        new_strat_names,
        new_payoffs
    )
    
    return reduced_game, log


# =============================================================================
# MAXMIN / MINMAX STRATEGIES
# =============================================================================

def maxmin_strategy(game: NormalFormGame, player: int) -> Tuple[np.ndarray, float]:
    """
    Find maxmin (security) strategy for a player in a 2-player game.
    
    The maxmin strategy maximizes the minimum expected payoff.
    
    Parameters
    ----------
    game : NormalFormGame
        A 2-player game
    player : int
        0 or 1
    
    Returns
    -------
    tuple of (mixed_strategy, maxmin_value)
    """
    if game.n_players != 2:
        raise ValueError("Only implemented for 2-player games")
    
    A = game.payoff_arrays[player]
    if player == 1:
        A = A.T  # Transpose so player is row player
    
    m, n = A.shape
    
    # LP formulation:
    # max v
    # s.t. A @ x >= v * 1  (for each column, expected payoff >= v)
    #      sum(x) = 1
    #      x >= 0
    
    # Rearrange: max v such that A @ x - v >= 0
    # Variables: [x_1, ..., x_m, v]
    
    # Objective: maximize v (minimize -v)
    c = np.zeros(m + 1)
    c[-1] = -1  # Minimize negative v = maximize v
    
    # Inequality constraints: A @ x - v >= 0  =>  -A @ x + v <= 0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    
    # Equality constraint: sum(x) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1])
    
    # Bounds: x >= 0, v unbounded
    bounds = [(0, 1) for _ in range(m)] + [(None, None)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                    bounds=bounds, method='highs')
    
    if result.success:
        strategy = result.x[:m]
        strategy = strategy / strategy.sum()  # Ensure normalization
        value = result.x[-1]
        return strategy, value
    else:
        # Fallback: uniform strategy
        return np.ones(m) / m, min(A.mean(axis=1))


def minmax_value(game: NormalFormGame, player: int) -> float:
    """
    Find the minmax value that opponents can hold player to.
    
    In zero-sum games: maxmin = minmax (minimax theorem)
    """
    if game.n_players != 2:
        raise ValueError("Only implemented for 2-player games")
    
    other = 1 - player
    A = game.payoff_arrays[player]
    if player == 1:
        A = A.T
    
    # Opponents minimize player's maximum payoff
    # min_q max_i (A @ q)_i
    
    m, n = A.shape
    
    # LP: min v such that A @ q <= v for all rows
    c = np.zeros(n + 1)
    c[-1] = 1
    
    A_ub = np.hstack([A, -np.ones((m, 1))])
    b_ub = np.zeros(m)
    
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1
    b_eq = np.array([1])
    
    bounds = [(0, 1) for _ in range(n)] + [(None, None)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='highs')
    
    if result.success:
        return result.x[-1]
    else:
        return float('nan')


# =============================================================================
# ZERO-SUM GAME SOLVER
# =============================================================================

def solve_zero_sum(A: np.ndarray) -> Dict:
    """
    Solve a two-player zero-sum game.
    
    Parameters
    ----------
    A : np.ndarray
        Payoff matrix for row player (column player gets -A)
    
    Returns
    -------
    dict with 'row_strategy', 'col_strategy', 'value'
    """
    m, n = A.shape
    
    # Shift matrix to ensure all positive (for LP)
    shift = abs(A.min()) + 1
    A_shifted = A + shift
    
    # Row player LP: max v s.t. A^T @ p >= v, sum(p) = 1
    c = np.zeros(m + 1)
    c[-1] = -1
    
    A_ub = np.hstack([-A_shifted.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1])
    
    bounds = [(0, 1) for _ in range(m)] + [(None, None)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method='highs')
    
    p = result.x[:m]
    p = p / p.sum()
    value = result.x[-1] - shift
    
    # Column player LP (dual)
    c_col = np.zeros(n + 1)
    c_col[-1] = 1
    
    A_ub_col = np.hstack([A_shifted, -np.ones((m, 1))])
    b_ub_col = np.zeros(m)
    
    A_eq_col = np.zeros((1, n + 1))
    A_eq_col[0, :n] = 1
    b_eq_col = np.array([1])
    
    bounds_col = [(0, 1) for _ in range(n)] + [(None, None)]
    
    result_col = linprog(c_col, A_ub=A_ub_col, b_ub=b_ub_col, 
                        A_eq=A_eq_col, b_eq=b_eq_col,
                        bounds=bounds_col, method='highs')
    
    q = result_col.x[:n]
    q = q / q.sum()
    
    return {
        'row_strategy': p,
        'col_strategy': q,
        'value': value
    }
