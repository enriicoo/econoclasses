"""
Classic games and economic applications.

Standard games from game theory textbooks plus
economic applications: Cournot, Bertrand, Stackelberg.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from .games import NormalFormGame, ExtensiveFormGame, GameNode


# =============================================================================
# CLASSIC 2x2 GAMES
# =============================================================================

def prisoners_dilemma(payoffs: Tuple[float, ...] = (-1, -3, 0, -2)) -> NormalFormGame:
    """
    The Prisoner's Dilemma.
    
    Default payoffs: CC=(-1,-1), CD=(-3,0), DC=(0,-3), DD=(-2,-2)
    
    Parameters
    ----------
    payoffs : tuple of 4 floats
        (CC, CD, DC, DD) payoffs for row player
        Column player gets transposed payoffs (symmetric game)
    
    Example
    -------
    >>> game = prisoners_dilemma()
    >>> print(game.payoff_table())
    """
    CC, CD, DC, DD = payoffs
    
    return NormalFormGame.from_payoff_dict(
        ['Row', 'Column'],
        [['Cooperate', 'Defect'], ['Cooperate', 'Defect']],
        {
            ('Cooperate', 'Cooperate'): (CC, CC),
            ('Cooperate', 'Defect'): (CD, DC),
            ('Defect', 'Cooperate'): (DC, CD),
            ('Defect', 'Defect'): (DD, DD),
        }
    )


def battle_of_sexes(his_pref: float = 3, her_pref: float = 2, 
                    mismatch: float = 0) -> NormalFormGame:
    """
    Battle of the Sexes coordination game.
    
    Two players must coordinate on an activity.
    Player 1 prefers Opera, Player 2 prefers Fight.
    
    Parameters
    ----------
    his_pref : float
        Payoff when at preferred event together
    her_pref : float  
        Payoff when at partner's preferred event together
    mismatch : float
        Payoff when at different events
    """
    return NormalFormGame.from_payoff_dict(
        ['Player 1', 'Player 2'],
        [['Opera', 'Fight'], ['Opera', 'Fight']],
        {
            ('Opera', 'Opera'): (his_pref, her_pref),
            ('Opera', 'Fight'): (mismatch, mismatch),
            ('Fight', 'Opera'): (mismatch, mismatch),
            ('Fight', 'Fight'): (her_pref, his_pref),
        }
    )


def matching_pennies() -> NormalFormGame:
    """
    Matching Pennies - a zero-sum game.
    
    Player 1 wins if pennies match, Player 2 wins if they differ.
    Classic example of game with no pure strategy Nash equilibrium.
    """
    return NormalFormGame.from_payoff_dict(
        ['Matcher', 'Mismatcher'],
        [['Heads', 'Tails'], ['Heads', 'Tails']],
        {
            ('Heads', 'Heads'): (1, -1),
            ('Heads', 'Tails'): (-1, 1),
            ('Tails', 'Heads'): (-1, 1),
            ('Tails', 'Tails'): (1, -1),
        }
    )


def stag_hunt(stag: float = 4, hare: float = 3, 
              nothing: float = 0) -> NormalFormGame:
    """
    Stag Hunt coordination game.
    
    Cooperation yields highest payoff but requires trust.
    Hunting hare alone is safe but suboptimal.
    
    Parameters
    ----------
    stag : float
        Payoff from successful stag hunt (requires both)
    hare : float
        Payoff from hunting hare alone
    nothing : float
        Payoff when hunting stag alone (partner hunts hare)
    """
    return NormalFormGame.from_payoff_dict(
        ['Hunter 1', 'Hunter 2'],
        [['Stag', 'Hare'], ['Stag', 'Hare']],
        {
            ('Stag', 'Stag'): (stag, stag),
            ('Stag', 'Hare'): (nothing, hare),
            ('Hare', 'Stag'): (hare, nothing),
            ('Hare', 'Hare'): (hare, hare),
        }
    )


def chicken(win: float = 7, tie: float = 6, lose: float = 2, 
            crash: float = 0) -> NormalFormGame:
    """
    Chicken (Hawk-Dove) game.
    
    Two players drive toward each other. Each can Swerve (chicken out) 
    or Go Straight. Mutual straight = crash.
    
    Parameters
    ----------
    win : float
        Payoff for going straight when other swerves
    tie : float
        Payoff when both swerve
    lose : float
        Payoff for swerving when other goes straight
    crash : float
        Payoff when both go straight (crash)
    """
    return NormalFormGame.from_payoff_dict(
        ['Driver 1', 'Driver 2'],
        [['Swerve', 'Straight'], ['Swerve', 'Straight']],
        {
            ('Swerve', 'Swerve'): (tie, tie),
            ('Swerve', 'Straight'): (lose, win),
            ('Straight', 'Swerve'): (win, lose),
            ('Straight', 'Straight'): (crash, crash),
        }
    )


def rock_paper_scissors() -> NormalFormGame:
    """
    Rock-Paper-Scissors - symmetric zero-sum game.
    
    Rock beats Scissors, Scissors beats Paper, Paper beats Rock.
    """
    return NormalFormGame.from_payoff_dict(
        ['Player 1', 'Player 2'],
        [['Rock', 'Paper', 'Scissors'], ['Rock', 'Paper', 'Scissors']],
        {
            ('Rock', 'Rock'): (0, 0),
            ('Rock', 'Paper'): (-1, 1),
            ('Rock', 'Scissors'): (1, -1),
            ('Paper', 'Rock'): (1, -1),
            ('Paper', 'Paper'): (0, 0),
            ('Paper', 'Scissors'): (-1, 1),
            ('Scissors', 'Rock'): (-1, 1),
            ('Scissors', 'Paper'): (1, -1),
            ('Scissors', 'Scissors'): (0, 0),
        }
    )


def pure_coordination(n: int = 2, coord_payoff: float = 1,
                      miscoord_payoff: float = 0) -> NormalFormGame:
    """
    Pure coordination game with n strategies.
    
    Players only get payoff if they choose the same strategy.
    """
    strategies = [f'S{i+1}' for i in range(n)]
    payoffs = {}
    
    for s1 in strategies:
        for s2 in strategies:
            if s1 == s2:
                payoffs[(s1, s2)] = (coord_payoff, coord_payoff)
            else:
                payoffs[(s1, s2)] = (miscoord_payoff, miscoord_payoff)
    
    return NormalFormGame.from_payoff_dict(
        ['Player 1', 'Player 2'],
        [strategies, strategies],
        payoffs
    )


# =============================================================================
# ECONOMIC APPLICATIONS: OLIGOPOLY
# =============================================================================

@dataclass
class CournotResult:
    """Result of Cournot duopoly analysis."""
    q1: float  # Firm 1 quantity
    q2: float  # Firm 2 quantity
    Q: float   # Total quantity
    P: float   # Market price
    profit1: float
    profit2: float
    consumer_surplus: float
    total_surplus: float


def cournot_duopoly(
    a: float = 100,         # Demand intercept
    b: float = 1,           # Demand slope
    c1: float = 10,         # Firm 1 marginal cost
    c2: float = 10,         # Firm 2 marginal cost
    F1: float = 0,          # Firm 1 fixed cost
    F2: float = 0           # Firm 2 fixed cost
) -> CournotResult:
    """
    Solve Cournot duopoly with linear demand and costs.
    
    Demand: P = a - b*Q
    Firm i cost: C_i = F_i + c_i * q_i
    
    Nash equilibrium: firms simultaneously choose quantities.
    
    Parameters
    ----------
    a : float
        Demand intercept (max willingness to pay)
    b : float
        Demand slope
    c1, c2 : float
        Marginal costs
    F1, F2 : float
        Fixed costs
    
    Returns
    -------
    CournotResult with equilibrium quantities, price, profits
    
    Example
    -------
    >>> result = cournot_duopoly(a=100, b=1, c1=10, c2=10)
    >>> print(f"q1={result.q1:.2f}, q2={result.q2:.2f}, P={result.P:.2f}")
    q1=30.00, q2=30.00, P=40.00
    """
    # Best response functions:
    # π1 = (a - b(q1 + q2))q1 - c1*q1 - F1
    # FOC: a - 2bq1 - bq2 - c1 = 0
    # BR1: q1 = (a - c1 - b*q2) / (2b)
    # Similarly BR2: q2 = (a - c2 - b*q1) / (2b)
    
    # Nash equilibrium (solve simultaneously):
    # q1 = (a - c1)/2b - q2/2
    # q2 = (a - c2)/2b - q1/2
    
    # Solution:
    # q1 = (a - 2c1 + c2) / (3b)
    # q2 = (a - 2c2 + c1) / (3b)
    
    q1 = max(0, (a - 2*c1 + c2) / (3*b))
    q2 = max(0, (a - 2*c2 + c1) / (3*b))
    Q = q1 + q2
    P = max(0, a - b*Q)
    
    profit1 = P * q1 - c1 * q1 - F1
    profit2 = P * q2 - c2 * q2 - F2
    
    # Consumer surplus: area under demand above price
    # CS = (1/2) * Q * (a - P) = (1/2) * Q * b * Q = (b/2) * Q^2
    CS = 0.5 * b * Q**2
    
    total_surplus = CS + profit1 + profit2
    
    return CournotResult(
        q1=q1, q2=q2, Q=Q, P=P,
        profit1=profit1, profit2=profit2,
        consumer_surplus=CS, total_surplus=total_surplus
    )


def cournot_best_response(a: float, b: float, c: float, 
                          q_other: float) -> float:
    """
    Cournot best response function.
    
    Given opponent's quantity, find profit-maximizing quantity.
    """
    return max(0, (a - c - b * q_other) / (2 * b))


def cournot_n_firms(
    a: float = 100,
    b: float = 1,
    costs: list = None,
    n: int = 2
) -> Dict:
    """
    Cournot oligopoly with n symmetric firms.
    
    Parameters
    ----------
    a : float
        Demand intercept
    b : float
        Demand slope  
    costs : list
        Marginal costs (if None, all = 10)
    n : int
        Number of firms (used if costs is None)
    
    Returns
    -------
    dict with quantities, price, profits
    """
    if costs is None:
        costs = [10.0] * n
    n = len(costs)
    
    # For symmetric firms with cost c:
    # q* = (a - c) / (b * (n + 1))
    
    # For asymmetric: solve system of FOCs
    # qi = (a - ci - b * sum(q_j, j≠i)) / (2b)
    
    # Matrix form: solve for q
    # 2b*q1 + b*q2 + ... + b*qn = a - c1
    # b*q1 + 2b*q2 + ... + b*qn = a - c2
    # etc.
    
    A = b * np.ones((n, n)) + b * np.eye(n)  # 2b on diagonal, b elsewhere
    rhs = np.array([a - c for c in costs])
    
    quantities = np.linalg.solve(A, rhs)
    quantities = np.maximum(quantities, 0)  # Non-negative
    
    Q = quantities.sum()
    P = max(0, a - b * Q)
    
    profits = [P * q - c * q for q, c in zip(quantities, costs)]
    CS = 0.5 * b * Q**2
    
    return {
        'quantities': quantities.tolist(),
        'total_quantity': Q,
        'price': P,
        'profits': profits,
        'consumer_surplus': CS,
        'n_firms': n
    }


@dataclass
class BertrandResult:
    """Result of Bertrand duopoly analysis."""
    p1: float  # Firm 1 price
    p2: float  # Firm 2 price
    q1: float  # Firm 1 quantity
    q2: float  # Firm 2 quantity
    profit1: float
    profit2: float
    equilibrium_type: str  # 'competitive', 'monopoly', 'asymmetric'


def bertrand_duopoly(
    a: float = 100,
    b: float = 1,
    c1: float = 10,
    c2: float = 10,
    differentiated: bool = False,
    d: float = 0.5  # Product differentiation parameter
) -> BertrandResult:
    """
    Solve Bertrand duopoly.
    
    Homogeneous products: Firms compete on price, P = min(MC) in equilibrium.
    Differentiated products: Prices above marginal cost.
    
    Parameters
    ----------
    a : float
        Demand intercept
    b : float
        Own-price sensitivity
    c1, c2 : float
        Marginal costs
    differentiated : bool
        If True, products are differentiated
    d : float
        Cross-price sensitivity (only if differentiated)
        Higher d = more substitutable = more competition
    
    Returns
    -------
    BertrandResult
    
    Example
    -------
    >>> result = bertrand_duopoly(a=100, b=1, c1=10, c2=10)
    >>> print(f"Prices: p1={result.p1:.2f}, p2={result.p2:.2f}")
    """
    if not differentiated:
        # Homogeneous Bertrand: price = min marginal cost
        p_eq = min(c1, c2)
        
        if c1 < c2:
            # Firm 1 captures market at price just below c2
            p1 = c2 - 0.01  # Epsilon below competitor's cost
            p2 = c2
            Q = max(0, a - b * p1)
            q1, q2 = Q, 0
            profit1 = (p1 - c1) * q1
            profit2 = 0
            eq_type = 'asymmetric'
        elif c2 < c1:
            p1 = c1
            p2 = c1 - 0.01
            Q = max(0, a - b * p2)
            q1, q2 = 0, Q
            profit1 = 0
            profit2 = (p2 - c2) * q2
            eq_type = 'asymmetric'
        else:
            # Equal costs: competitive equilibrium
            p1 = p2 = c1
            Q = max(0, a - b * c1)
            q1 = q2 = Q / 2
            profit1 = profit2 = 0
            eq_type = 'competitive'
        
        return BertrandResult(
            p1=p1, p2=p2, q1=q1, q2=q2,
            profit1=profit1, profit2=profit2,
            equilibrium_type=eq_type
        )
    
    else:
        # Differentiated Bertrand
        # Demand: qi = a - b*pi + d*pj
        # Profit: πi = (pi - ci)(a - b*pi + d*pj)
        # FOC: a - 2b*pi + d*pj + b*ci = 0
        # BR: pi = (a + b*ci + d*pj) / (2b)
        
        # Solve system:
        # 2b*p1 - d*p2 = a + b*c1
        # -d*p1 + 2b*p2 = a + b*c2
        
        A = np.array([[2*b, -d], [-d, 2*b]])
        rhs = np.array([a + b*c1, a + b*c2])
        
        prices = np.linalg.solve(A, rhs)
        p1, p2 = prices
        
        q1 = max(0, a - b*p1 + d*p2)
        q2 = max(0, a - b*p2 + d*p1)
        
        profit1 = (p1 - c1) * q1
        profit2 = (p2 - c2) * q2
        
        return BertrandResult(
            p1=p1, p2=p2, q1=q1, q2=q2,
            profit1=profit1, profit2=profit2,
            equilibrium_type='differentiated'
        )


@dataclass
class StackelbergResult:
    """Result of Stackelberg duopoly analysis."""
    q_leader: float
    q_follower: float
    Q: float
    P: float
    profit_leader: float
    profit_follower: float


def stackelberg_duopoly(
    a: float = 100,
    b: float = 1,
    c_leader: float = 10,
    c_follower: float = 10
) -> StackelbergResult:
    """
    Solve Stackelberg duopoly with quantity leadership.
    
    Leader moves first, anticipating follower's best response.
    
    Parameters
    ----------
    a : float
        Demand intercept
    b : float
        Demand slope
    c_leader, c_follower : float
        Marginal costs
    
    Returns
    -------
    StackelbergResult
    
    Example
    -------
    >>> result = stackelberg_duopoly(a=100, b=1, c_leader=10, c_follower=10)
    >>> print(f"Leader: q={result.q_leader:.2f}, Follower: q={result.q_follower:.2f}")
    """
    # Follower's best response: qF = (a - cF - b*qL) / (2b)
    # Leader maximizes: πL = (a - b*(qL + qF) - cL) * qL
    # Substitute BR: πL = (a - b*qL - b*(a - cF - b*qL)/(2b) - cL) * qL
    #                   = (a - cL - b*qL - (a - cF - b*qL)/2) * qL
    #                   = ((a - 2cL + cF)/2 - b*qL/2) * qL
    # FOC: (a - 2cL + cF)/2 - b*qL = 0
    # qL* = (a - 2cL + cF) / (2b)
    
    q_leader = max(0, (a - 2*c_leader + c_follower) / (2*b))
    q_follower = max(0, (a - c_follower - b*q_leader) / (2*b))
    
    Q = q_leader + q_follower
    P = max(0, a - b*Q)
    
    profit_leader = (P - c_leader) * q_leader
    profit_follower = (P - c_follower) * q_follower
    
    return StackelbergResult(
        q_leader=q_leader, q_follower=q_follower,
        Q=Q, P=P,
        profit_leader=profit_leader, profit_follower=profit_follower
    )


# =============================================================================
# EXTENSIVE FORM GAME EXAMPLES
# =============================================================================

def entry_game(entrant_payoff_enter_fight: float = -1,
               entrant_payoff_enter_accommodate: float = 2,
               entrant_payoff_out: float = 0,
               incumbent_payoff_fight: float = -1,
               incumbent_payoff_accommodate: float = 1,
               incumbent_payoff_no_entry: float = 4) -> ExtensiveFormGame:
    """
    Market entry game.
    
    1. Entrant decides: Enter or Stay Out
    2. If Enter, Incumbent decides: Fight or Accommodate
    
    Returns
    -------
    ExtensiveFormGame
    """
    root = GameNode("entry_decision", player=0)
    root.add_child("Out", GameNode("out", 
                   payoffs=(entrant_payoff_out, incumbent_payoff_no_entry)))
    
    inc_decision = root.add_child("Enter", GameNode("inc_decision", player=1))
    inc_decision.add_child("Fight", GameNode("fight", 
                          payoffs=(entrant_payoff_enter_fight, incumbent_payoff_fight)))
    inc_decision.add_child("Accommodate", GameNode("accommodate", 
                          payoffs=(entrant_payoff_enter_accommodate, incumbent_payoff_accommodate)))
    
    return ExtensiveFormGame(root, ["Entrant", "Incumbent"])


def ultimatum_game(pie: float = 10, 
                   offers: list = None) -> ExtensiveFormGame:
    """
    Ultimatum bargaining game.
    
    1. Proposer offers a split of the pie
    2. Responder accepts or rejects
    
    Parameters
    ----------
    pie : float
        Total amount to split
    offers : list
        Possible offer amounts (default: [0, 2, 4, 5, 6, 8, 10])
    """
    if offers is None:
        offers = [0, 2, 4, 5, 6, 8, 10]
    
    root = GameNode("propose", player=0)
    
    for offer in offers:
        offer_node = root.add_child(
            f"offer_{offer}", 
            GameNode(f"respond_{offer}", player=1)
        )
        offer_node.add_child(
            "Accept",
            GameNode(f"accept_{offer}", payoffs=(pie - offer, offer))
        )
        offer_node.add_child(
            "Reject", 
            GameNode(f"reject_{offer}", payoffs=(0, 0))
        )
    
    return ExtensiveFormGame(root, ["Proposer", "Responder"])


def centipede_game(n_rounds: int = 4, 
                   grow_factor: float = 2) -> ExtensiveFormGame:
    """
    Centipede game.
    
    Players alternate. Each can Take (end game) or Pass (continue).
    Pot grows each round. 
    
    Parameters
    ----------
    n_rounds : int
        Number of decision nodes
    grow_factor : float
        Multiplier for pot each round
    """
    pot = 1.0
    
    root = GameNode("round_1", player=0)
    current = root
    
    for i in range(n_rounds):
        player = i % 2
        pot_share = pot / 2
        
        # Take: current player gets more
        if player == 0:
            take_payoffs = (pot_share * 1.5, pot_share * 0.5)
        else:
            take_payoffs = (pot_share * 0.5, pot_share * 1.5)
        
        current.add_child("Take", GameNode(f"take_{i+1}", payoffs=take_payoffs))
        
        if i < n_rounds - 1:
            # Pass: continue to next round
            pot *= grow_factor
            next_node = GameNode(f"round_{i+2}", player=1 - player)
            current.add_child("Pass", next_node)
            current = next_node
        else:
            # Final round: pass gives equal split
            final_split = (pot / 2, pot / 2)
            current.add_child("Pass", GameNode("final", payoffs=final_split))
    
    return ExtensiveFormGame(root, ["Player 1", "Player 2"])


# =============================================================================
# GAME COMPARISON AND ANALYSIS
# =============================================================================

def compare_oligopoly_models(a: float = 100, b: float = 1, 
                              c: float = 10) -> Dict:
    """
    Compare Cournot, Bertrand, and Stackelberg outcomes.
    
    Parameters
    ----------
    a, b, c : float
        Demand and cost parameters (symmetric firms)
    
    Returns
    -------
    dict comparing equilibrium outcomes
    """
    cournot = cournot_duopoly(a=a, b=b, c1=c, c2=c)
    bertrand = bertrand_duopoly(a=a, b=b, c1=c, c2=c)
    stackelberg = stackelberg_duopoly(a=a, b=b, c_leader=c, c_follower=c)
    
    # Perfect competition and monopoly benchmarks
    Q_competitive = (a - c) / b
    P_competitive = c
    
    Q_monopoly = (a - c) / (2 * b)
    P_monopoly = (a + c) / 2
    profit_monopoly = (a - c)**2 / (4 * b)
    
    return {
        'cournot': {
            'quantities': (cournot.q1, cournot.q2),
            'total_Q': cournot.Q,
            'price': cournot.P,
            'profits': (cournot.profit1, cournot.profit2),
            'total_profit': cournot.profit1 + cournot.profit2,
            'consumer_surplus': cournot.consumer_surplus
        },
        'bertrand': {
            'prices': (bertrand.p1, bertrand.p2),
            'quantities': (bertrand.q1, bertrand.q2),
            'total_Q': bertrand.q1 + bertrand.q2,
            'profits': (bertrand.profit1, bertrand.profit2),
            'total_profit': bertrand.profit1 + bertrand.profit2
        },
        'stackelberg': {
            'quantities': (stackelberg.q_leader, stackelberg.q_follower),
            'total_Q': stackelberg.Q,
            'price': stackelberg.P,
            'profits': (stackelberg.profit_leader, stackelberg.profit_follower),
            'total_profit': stackelberg.profit_leader + stackelberg.profit_follower
        },
        'monopoly': {
            'quantity': Q_monopoly,
            'price': P_monopoly,
            'profit': profit_monopoly
        },
        'competitive': {
            'quantity': Q_competitive,
            'price': P_competitive
        },
        'parameters': {'a': a, 'b': b, 'c': c}
    }
