"""
Advanced Game Theory
====================

Repeated games, Bayesian games, subgame perfect equilibrium, 
and mechanism design fundamentals.

Author: econoclasses
Version: 0.5.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict, Union
from scipy.optimize import minimize, brentq


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RepeatedGameResult:
    """Results from repeated game analysis."""
    stage_game_payoffs: Tuple[float, float]
    discount_factor: float
    cooperation_sustainable: bool
    critical_discount: float
    cooperation_payoffs: Tuple[float, float]
    defection_payoffs: Tuple[float, float]
    punishment_payoffs: Tuple[float, float]
    strategy_description: str
    
    def __repr__(self):
        return (f"RepeatedGameResult\n"
                f"  Stage game: {self.stage_game_payoffs}\n"
                f"  δ = {self.discount_factor:.3f}, δ* = {self.critical_discount:.3f}\n"
                f"  Cooperation sustainable: {self.cooperation_sustainable}\n"
                f"  Strategy: {self.strategy_description}")


@dataclass
class SubgamePerfectResult:
    """Results from subgame perfect equilibrium analysis."""
    equilibrium_path: List[str]
    equilibrium_payoffs: Tuple[float, ...]
    backward_induction_steps: List[Dict]
    is_unique: bool
    credible_threats: List[str]
    
    def __repr__(self):
        return (f"SubgamePerfectEquilibrium\n"
                f"  Path: {' → '.join(self.equilibrium_path)}\n"
                f"  Payoffs: {self.equilibrium_payoffs}\n"
                f"  Unique: {self.is_unique}")


@dataclass
class BayesianGameResult:
    """Results from Bayesian Nash equilibrium analysis."""
    player_types: Dict[int, List[str]]
    prior_beliefs: Dict[int, Dict[str, float]]
    strategies: Dict[int, Dict[str, str]]  # type -> action
    expected_payoffs: Dict[int, float]
    interim_payoffs: Dict[int, Dict[str, float]]  # player -> {type: payoff}
    
    def __repr__(self):
        strat_str = "\n".join(f"  Player {p}: {s}" for p, s in self.strategies.items())
        return (f"BayesianNashEquilibrium\n"
                f"Strategies:\n{strat_str}\n"
                f"  Expected payoffs: {self.expected_payoffs}")


@dataclass
class MechanismResult:
    """Results from mechanism design analysis."""
    mechanism_name: str
    allocation_rule: str
    payment_rule: str
    is_incentive_compatible: bool
    is_individually_rational: bool
    expected_revenue: float
    expected_efficiency: float
    description: str
    
    def __repr__(self):
        return (f"MechanismResult({self.mechanism_name})\n"
                f"  IC: {self.is_incentive_compatible}, IR: {self.is_individually_rational}\n"
                f"  E[Revenue] = {self.expected_revenue:.2f}\n"
                f"  E[Efficiency] = {self.expected_efficiency:.2f}")


# ============================================================================
# REPEATED GAMES
# ============================================================================

def analyze_repeated_game(
    cooperate_payoffs: Tuple[float, float],
    defect_payoffs: Tuple[float, float],
    punishment_payoffs: Tuple[float, float],
    sucker_payoffs: Tuple[float, float],
    discount_factor: float,
    strategy: str = "grim_trigger"
) -> RepeatedGameResult:
    """
    Analyze infinitely repeated game with discounting.
    
    Standard payoff structure (Prisoner's Dilemma style):
    - Both cooperate: cooperate_payoffs (R, R) - Reward
    - Both defect: punishment_payoffs (P, P) - Punishment  
    - I defect, you cooperate: defect_payoffs (T, S) - Temptation, Sucker
    - I cooperate, you defect: sucker_payoffs (S, T)
    
    For cooperation to be sustainable with grim trigger:
    δ ≥ (T - R) / (T - P)
    
    Parameters
    ----------
    cooperate_payoffs : Tuple[float, float]
        (R, R) payoffs when both cooperate
    defect_payoffs : Tuple[float, float]
        (T, S) payoffs when player 1 defects, player 2 cooperates
    punishment_payoffs : Tuple[float, float]
        (P, P) payoffs in Nash equilibrium (both defect)
    sucker_payoffs : Tuple[float, float]
        (S, T) payoffs when player 1 cooperates, player 2 defects
    discount_factor : float
        δ in (0, 1), patience parameter
    strategy : str
        'grim_trigger', 'tit_for_tat', or 'nash_reversion'
        
    Returns
    -------
    RepeatedGameResult
        Analysis of whether cooperation is sustainable
    """
    R = cooperate_payoffs[0]  # Reward for mutual cooperation
    T = defect_payoffs[0]      # Temptation to defect
    P = punishment_payoffs[0]  # Punishment (Nash)
    S = sucker_payoffs[0]      # Sucker's payoff
    
    delta = discount_factor
    
    # Critical discount factor for grim trigger
    # V(cooperate) = R + δR + δ²R + ... = R/(1-δ)
    # V(defect) = T + δP + δ²P + ... = T + δP/(1-δ)
    # Cooperation if: R/(1-δ) ≥ T + δP/(1-δ)
    # R ≥ T(1-δ) + δP
    # R ≥ T - δT + δP
    # δ(T-P) ≥ T - R
    # δ ≥ (T-R)/(T-P)
    
    if T > P:  # Normal case
        critical_delta = (T - R) / (T - P)
    else:
        critical_delta = 0.0  # Cooperation always sustainable
    
    cooperation_sustainable = delta >= critical_delta
    
    strategy_desc = {
        'grim_trigger': "Cooperate until defection, then defect forever",
        'tit_for_tat': "Start cooperate, then copy opponent's last move",
        'nash_reversion': "Cooperate, revert to Nash equilibrium after defection"
    }.get(strategy, strategy)
    
    return RepeatedGameResult(
        stage_game_payoffs=(R, R),
        discount_factor=delta,
        cooperation_sustainable=cooperation_sustainable,
        critical_discount=max(0, min(1, critical_delta)),
        cooperation_payoffs=cooperate_payoffs,
        defection_payoffs=defect_payoffs,
        punishment_payoffs=punishment_payoffs,
        strategy_description=strategy_desc
    )


def folk_theorem_payoffs(
    cooperate_payoffs: Tuple[float, float],
    defect_payoffs: Tuple[float, float],
    punishment_payoffs: Tuple[float, float],
    sucker_payoffs: Tuple[float, float],
    discount_factor: float = 0.9,
    num_points: int = 50
) -> Dict:
    """
    Compute the set of feasible and individually rational payoffs
    (Folk Theorem characterization).
    
    Returns
    -------
    Dict with:
        - feasible_set: Convex hull of stage game payoffs
        - ir_payoffs: Individually rational threshold (minmax)
        - sustainable_payoffs: Achievable payoffs for given δ
    """
    # Payoff matrix corners
    payoffs = np.array([
        cooperate_payoffs,      # (C, C)
        defect_payoffs,         # (D, C)  
        sucker_payoffs,         # (C, D)
        punishment_payoffs      # (D, D)
    ])
    
    # Minmax values (individually rational threshold)
    # In 2x2 game, minmax is typically the Nash payoff
    minmax_1 = punishment_payoffs[0]
    minmax_2 = punishment_payoffs[1]
    
    # Compute convex hull (feasible payoffs)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(payoffs)
        feasible_vertices = payoffs[hull.vertices]
    except:
        feasible_vertices = payoffs
    
    return {
        'feasible_vertices': feasible_vertices.tolist(),
        'minmax_payoffs': (minmax_1, minmax_2),
        'discount_factor': discount_factor,
        'folk_theorem_applies': discount_factor > 0.5,
        'description': (
            f"Folk Theorem: For δ sufficiently close to 1, any feasible payoff "
            f"above ({minmax_1:.2f}, {minmax_2:.2f}) is achievable in equilibrium."
        )
    }


def repeated_cournot(
    a: float = 100,
    c: float = 10,
    n_firms: int = 2,
    discount_factor: float = 0.9
) -> Dict:
    """
    Analyze repeated Cournot oligopoly with potential for collusion.
    
    Firms may sustain monopoly/cartel outcomes through repeated interaction.
    
    Parameters
    ----------
    a : float
        Demand intercept (P = a - Q)
    c : float
        Constant marginal cost
    n_firms : int
        Number of firms
    discount_factor : float
        Common discount factor δ
        
    Returns
    -------
    Dict with competitive, collusive, and deviation analysis
    """
    # Competitive (Nash) Cournot outcome
    q_nash = (a - c) / (n_firms + 1)
    Q_nash = n_firms * q_nash
    P_nash = a - Q_nash
    profit_nash = (P_nash - c) * q_nash
    
    # Collusive (joint monopoly) outcome
    Q_monopoly = (a - c) / 2
    q_collusive = Q_monopoly / n_firms
    P_monopoly = a - Q_monopoly
    profit_collusive = (P_monopoly - c) * q_collusive
    
    # Deviation profit (best response to collusion)
    # If others produce q_collusive each, my BR:
    # max_q (a - q - (n-1)*q_collusive - c) * q
    # FOC: a - 2q - (n-1)*q_collusive - c = 0
    # q_dev = (a - c - (n-1)*q_collusive) / 2
    q_deviation = (a - c - (n_firms - 1) * q_collusive) / 2
    Q_with_deviation = q_deviation + (n_firms - 1) * q_collusive
    P_deviation = a - Q_with_deviation
    profit_deviation = (P_deviation - c) * q_deviation
    
    # Critical discount factor for sustaining collusion
    # V(collude) = π_c / (1-δ)
    # V(deviate) = π_d + δ * π_nash / (1-δ)
    # Collusion if: π_c / (1-δ) ≥ π_d + δ * π_nash / (1-δ)
    # π_c ≥ π_d(1-δ) + δ * π_nash
    # π_c - π_nash*δ ≥ π_d - π_d*δ
    # π_c - δ(π_nash - π_d + π_d) ≥ π_d - π_d*δ... 
    # Simplified: δ* = (π_d - π_c) / (π_d - π_nash)
    
    if profit_deviation > profit_nash:
        critical_delta = (profit_deviation - profit_collusive) / (profit_deviation - profit_nash)
    else:
        critical_delta = 0.0
    
    collusion_sustainable = discount_factor >= critical_delta
    
    return {
        'nash_outcome': {
            'quantity_per_firm': q_nash,
            'total_quantity': Q_nash,
            'price': P_nash,
            'profit_per_firm': profit_nash
        },
        'collusive_outcome': {
            'quantity_per_firm': q_collusive,
            'total_quantity': Q_monopoly,
            'price': P_monopoly,
            'profit_per_firm': profit_collusive
        },
        'deviation_outcome': {
            'deviator_quantity': q_deviation,
            'deviator_profit': profit_deviation
        },
        'critical_discount_factor': max(0, min(1, critical_delta)),
        'discount_factor': discount_factor,
        'collusion_sustainable': collusion_sustainable,
        'profit_gain_from_collusion': profit_collusive - profit_nash,
        'description': (
            f"With δ={discount_factor:.2f}, collusion is "
            f"{'sustainable' if collusion_sustainable else 'NOT sustainable'} "
            f"(need δ ≥ {critical_delta:.3f})"
        )
    }


# ============================================================================
# BAYESIAN GAMES (INCOMPLETE INFORMATION)
# ============================================================================

def bayesian_game_2x2(
    payoff_matrix_high: np.ndarray,
    payoff_matrix_low: np.ndarray,
    prob_high: float,
    informed_player: int = 2
) -> BayesianGameResult:
    """
    Solve a simple 2x2 Bayesian game where one player has private information.
    
    Player 2 knows their type (high or low), Player 1 doesn't.
    
    Parameters
    ----------
    payoff_matrix_high : np.ndarray
        2x2x2 payoff matrix when player 2 is high type
        payoff_matrix[i,j] = (payoff_1, payoff_2) for action profile (i,j)
    payoff_matrix_low : np.ndarray
        2x2x2 payoff matrix when player 2 is low type
    prob_high : float
        Prior probability that player 2 is high type
    informed_player : int
        Which player has private info (default: 2)
        
    Returns
    -------
    BayesianGameResult
        Bayesian Nash equilibrium
    """
    # Simple case: informed player has dominant strategy for each type
    # Player 1 best responds to expected payoffs
    
    prob_low = 1 - prob_high
    
    # Check if player 2 has dominant strategies by type
    # High type
    high_payoffs_2 = payoff_matrix_high[:, :, 1]  # Player 2's payoffs
    high_type_strategy = None
    if np.all(high_payoffs_2[:, 0] >= high_payoffs_2[:, 1]):
        high_type_strategy = 'A'  # Action 0
    elif np.all(high_payoffs_2[:, 1] >= high_payoffs_2[:, 0]):
        high_type_strategy = 'B'  # Action 1
    
    # Low type
    low_payoffs_2 = payoff_matrix_low[:, :, 1]
    low_type_strategy = None
    if np.all(low_payoffs_2[:, 0] >= low_payoffs_2[:, 1]):
        low_type_strategy = 'A'
    elif np.all(low_payoffs_2[:, 1] >= low_payoffs_2[:, 0]):
        low_type_strategy = 'B'
    
    # Player 1's expected payoffs
    # Assume separating equilibrium if strategies differ
    expected_payoffs_1 = np.zeros(2)
    for a1 in range(2):
        if high_type_strategy == 'A':
            a2_high = 0
        elif high_type_strategy == 'B':
            a2_high = 1
        else:
            a2_high = 0  # Default
            
        if low_type_strategy == 'A':
            a2_low = 0
        elif low_type_strategy == 'B':
            a2_low = 1
        else:
            a2_low = 0
            
        expected_payoffs_1[a1] = (
            prob_high * payoff_matrix_high[a1, a2_high, 0] +
            prob_low * payoff_matrix_low[a1, a2_low, 0]
        )
    
    player_1_strategy = 'A' if expected_payoffs_1[0] >= expected_payoffs_1[1] else 'B'
    
    return BayesianGameResult(
        player_types={1: ['uninformed'], 2: ['high', 'low']},
        prior_beliefs={1: {'high': prob_high, 'low': prob_low}},
        strategies={
            1: {'uninformed': player_1_strategy},
            2: {'high': high_type_strategy or 'A', 'low': low_type_strategy or 'A'}
        },
        expected_payoffs={1: expected_payoffs_1.max(), 2: 0.0},  # Simplified
        interim_payoffs={1: {'uninformed': expected_payoffs_1.max()}, 2: {'high': 0.0, 'low': 0.0}}
    )


def signaling_game(
    sender_types: List[str],
    type_priors: Dict[str, float],
    signals: List[str],
    signal_costs: Dict[str, Dict[str, float]],
    receiver_actions: List[str],
    payoffs: Dict  # Complex structure
) -> Dict:
    """
    Analyze a signaling game (e.g., Spence job market signaling).
    
    Parameters
    ----------
    sender_types : List[str]
        Possible types of sender (e.g., ['high', 'low'])
    type_priors : Dict[str, float]
        Prior probabilities of each type
    signals : List[str]
        Available signals (e.g., ['educate', 'no_educate'])
    signal_costs : Dict[str, Dict[str, float]]
        Cost of each signal for each type
    receiver_actions : List[str]
        Receiver's actions (e.g., ['hire_high', 'hire_low'])
    payoffs : Dict
        Payoff structure
        
    Returns
    -------
    Dict with separating and pooling equilibrium analysis
    """
    # Simplified Spence signaling model
    # High type: cost of education = c_h
    # Low type: cost of education = c_l > c_h (single crossing)
    
    c_h = signal_costs.get('high', {}).get('educate', 1)
    c_l = signal_costs.get('low', {}).get('educate', 2)
    
    w_h = payoffs.get('high_wage', 100)
    w_l = payoffs.get('low_wage', 50)
    
    # Separating equilibrium conditions:
    # High type: w_h - c_h ≥ w_l (prefers to signal)
    # Low type: w_l ≥ w_h - c_l (prefers not to signal)
    
    high_wants_to_signal = (w_h - c_h) >= w_l
    low_wants_to_not_signal = w_l >= (w_h - c_l)
    
    separating_exists = high_wants_to_signal and low_wants_to_not_signal
    
    # Pooling equilibrium (both types same signal)
    # Can exist if beliefs off-path support it
    
    return {
        'separating_equilibrium': {
            'exists': separating_exists,
            'high_type_strategy': 'educate' if separating_exists else 'depends',
            'low_type_strategy': 'no_educate' if separating_exists else 'depends',
            'high_type_payoff': w_h - c_h if separating_exists else None,
            'low_type_payoff': w_l if separating_exists else None
        },
        'pooling_equilibrium': {
            'exists': True,  # Often exists with appropriate off-path beliefs
            'both_educate': {
                'expected_wage': type_priors['high'] * w_h + type_priors['low'] * w_l
            }
        },
        'single_crossing': c_l > c_h,
        'description': (
            f"Separating equilibrium {'exists' if separating_exists else 'does not exist'}. "
            f"Single crossing: {c_l > c_h}"
        )
    }


# ============================================================================
# SUBGAME PERFECT EQUILIBRIUM
# ============================================================================

def backward_induction(
    game_tree: Dict,
    start_node: str = 'root'
) -> SubgamePerfectResult:
    """
    Solve extensive form game by backward induction.
    
    Parameters
    ----------
    game_tree : Dict
        Tree structure with nodes, players, actions, and payoffs
        Format: {
            'node_name': {
                'player': int or 'terminal',
                'actions': {'action': 'next_node', ...} or None,
                'payoffs': (p1, p2, ...) if terminal else None
            }
        }
    start_node : str
        Root node name
        
    Returns
    -------
    SubgamePerfectResult
        SPE path and payoffs
    """
    # Simple recursive backward induction
    def solve_node(node_name):
        node = game_tree[node_name]
        
        if node['player'] == 'terminal':
            return node['payoffs'], []
        
        player = node['player']
        best_action = None
        best_payoff = None
        best_path = None
        
        for action, next_node in node['actions'].items():
            payoffs, path = solve_node(next_node)
            
            if best_payoff is None or payoffs[player - 1] > best_payoff:
                best_payoff = payoffs[player - 1]
                best_action = action
                best_path = [f"P{player}:{action}"] + path
                final_payoffs = payoffs
        
        return final_payoffs, best_path
    
    payoffs, path = solve_node(start_node)
    
    return SubgamePerfectResult(
        equilibrium_path=path,
        equilibrium_payoffs=payoffs,
        backward_induction_steps=[],  # Could add detailed steps
        is_unique=True,  # Assuming generic payoffs
        credible_threats=[]
    )


def centipede_game_analysis(n_rounds: int = 6, pot_growth: float = 2.0) -> Dict:
    """
    Analyze the centipede game with backward induction.
    
    Two players alternate, can 'continue' (grow the pot) or 'stop' (take most of pot).
    Backward induction predicts stop at first node, but experiments show continuation.
    
    Parameters
    ----------
    n_rounds : int
        Number of decision nodes
    pot_growth : float
        Multiplier when continuing
        
    Returns
    -------
    Dict with theoretical vs. experimental analysis
    """
    # Build payoffs
    # At round r, if stopped: (larger_share, smaller_share)
    payoffs = []
    for r in range(n_rounds + 1):
        pot = pot_growth ** r
        larger = pot * 0.6
        smaller = pot * 0.4
        # Odd rounds: player 1's turn (0, 2, 4...)
        # Player who stops gets larger share
        payoffs.append((larger, smaller) if r % 2 == 0 else (smaller, larger))
    
    # Backward induction says stop immediately
    spe_stop_round = 0
    spe_payoffs = payoffs[0]
    
    # But continuation to end gives much more
    continuation_payoffs = payoffs[-1]
    
    return {
        'n_rounds': n_rounds,
        'spe_prediction': {
            'stop_at': spe_stop_round,
            'payoffs': spe_payoffs
        },
        'full_cooperation': {
            'stop_at': n_rounds,
            'payoffs': continuation_payoffs
        },
        'efficiency_loss': 1 - sum(spe_payoffs) / sum(continuation_payoffs),
        'experimental_findings': (
            "In experiments, players often continue for several rounds, "
            "suggesting bounded rationality or altruism."
        ),
        'paradox': (
            "Backward induction predicts immediate stopping, but mutual "
            "continuation is Pareto superior. This illustrates tension between "
            "individual and collective rationality."
        )
    }


# ============================================================================
# MECHANISM DESIGN BASICS
# ============================================================================

def revelation_principle_example() -> Dict:
    """
    Illustrate the Revelation Principle with a simple example.
    
    Any outcome achievable by any mechanism can be achieved by a 
    direct mechanism where agents truthfully report their types.
    
    Returns
    -------
    Dict with example and explanation
    """
    return {
        'principle': (
            "If a social choice function f can be implemented by ANY mechanism, "
            "then it can be implemented by a DIRECT mechanism where: "
            "(1) agents report types, (2) mechanism chooses f(reports), "
            "(3) truthful reporting is a Bayesian Nash equilibrium."
        ),
        'implication': (
            "We can restrict attention to direct, incentive-compatible mechanisms "
            "without loss of generality."
        ),
        'example': {
            'scenario': "Allocating a single good to one of two buyers",
            'indirect_mechanism': "English auction",
            'direct_mechanism': "Sealed-bid second-price auction (Vickrey)",
            'equivalence': "Both achieve same allocation with truthful bidding as equilibrium"
        }
    }


def vickrey_clarke_groves(
    valuations: Dict[str, Dict[str, float]],
    feasible_allocations: List[str]
) -> MechanismResult:
    """
    VCG mechanism for efficient allocation with private values.
    
    Each agent pays their externality on others.
    
    Parameters
    ----------
    valuations : Dict[str, Dict[str, float]]
        {agent: {allocation: value}}
    feasible_allocations : List[str]
        Possible allocation choices
        
    Returns
    -------
    MechanismResult
        VCG outcome
    """
    agents = list(valuations.keys())
    
    # Find efficient allocation (maximize total value)
    best_allocation = None
    best_total = -float('inf')
    
    for alloc in feasible_allocations:
        total = sum(valuations[a].get(alloc, 0) for a in agents)
        if total > best_total:
            best_total = total
            best_allocation = alloc
    
    # Compute VCG payments
    # Payment_i = max welfare of others without i - welfare of others with i
    payments = {}
    for agent in agents:
        others = [a for a in agents if a != agent]
        
        # Best allocation without agent i
        best_without_i = -float('inf')
        for alloc in feasible_allocations:
            total_others = sum(valuations[a].get(alloc, 0) for a in others)
            if total_others > best_without_i:
                best_without_i = total_others
        
        # Welfare of others under chosen allocation
        welfare_others_with_i = sum(valuations[a].get(best_allocation, 0) for a in others)
        
        payments[agent] = best_without_i - welfare_others_with_i
    
    total_revenue = sum(payments.values())
    
    return MechanismResult(
        mechanism_name="VCG (Vickrey-Clarke-Groves)",
        allocation_rule=f"Choose allocation maximizing Σv_i: {best_allocation}",
        payment_rule="Each agent pays their externality on others",
        is_incentive_compatible=True,
        is_individually_rational=True,  # With proper transfers
        expected_revenue=total_revenue,
        expected_efficiency=1.0,  # VCG is always efficient
        description=f"Payments: {payments}"
    )


def myerson_optimal_auction(
    value_distributions: Dict[str, Tuple[float, float]],
    reserve_price: Optional[float] = None
) -> MechanismResult:
    """
    Myerson's optimal auction for revenue maximization.
    
    With symmetric bidders and uniform distributions, this reduces to 
    second-price auction with optimal reserve.
    
    Parameters
    ----------
    value_distributions : Dict[str, Tuple[float, float]]
        {bidder: (low, high)} for uniform distributions
    reserve_price : Optional[float]
        If None, computes optimal reserve
        
    Returns
    -------
    MechanismResult
        Optimal auction design
    """
    # Simplified: assume symmetric uniform bidders
    bidders = list(value_distributions.keys())
    n = len(bidders)
    
    # Assume all have same distribution
    v_low, v_high = list(value_distributions.values())[0]
    
    # Virtual value: φ(v) = v - (1-F(v))/f(v)
    # For uniform [a,b]: F(v) = (v-a)/(b-a), f(v) = 1/(b-a)
    # φ(v) = v - (b-v)/1 = 2v - b
    
    # Optimal reserve: φ(r) = 0 => r = b/2 (when a=0)
    if reserve_price is None:
        optimal_reserve = (v_low + v_high) / 2
    else:
        optimal_reserve = reserve_price
    
    # Expected revenue formula for 2 bidders, uniform [0,1]
    # E[Revenue] = (n-1)/(n+1) + r*(1-r)^n / [1 - (1-r)^n] (complex formula)
    # Simplified: with optimal reserve r=0.5, E[Rev] ≈ 5/12 for n=2
    if n == 2 and v_low == 0 and v_high == 1:
        expected_revenue = 5/12  # Approximately
    else:
        # Rough approximation
        expected_revenue = (n-1)/(n+1) * (v_high - v_low) + v_low
    
    return MechanismResult(
        mechanism_name="Myerson Optimal Auction",
        allocation_rule="Give to highest virtual value if > 0",
        payment_rule="Charge minimum bid to win (with reserve)",
        is_incentive_compatible=True,
        is_individually_rational=True,
        expected_revenue=expected_revenue,
        expected_efficiency=0.95,  # Efficient except when reserve binds
        description=f"Optimal reserve price: {optimal_reserve:.2f}"
    )


# ============================================================================
# EXAMPLE APPLICATIONS
# ============================================================================

def example_repeated_prisoners_dilemma():
    """
    Classic repeated Prisoner's Dilemma analysis.
    
    Returns analysis of when cooperation can be sustained.
    """
    # Standard payoffs: T > R > P > S
    # R = 3 (both cooperate)
    # T = 5 (I defect, you cooperate)  
    # P = 1 (both defect)
    # S = 0 (I cooperate, you defect)
    
    results = {}
    
    for delta in [0.3, 0.5, 0.7, 0.9]:
        result = analyze_repeated_game(
            cooperate_payoffs=(3, 3),
            defect_payoffs=(5, 0),
            punishment_payoffs=(1, 1),
            sucker_payoffs=(0, 5),
            discount_factor=delta
        )
        results[f"delta={delta}"] = {
            'sustainable': result.cooperation_sustainable,
            'critical_delta': result.critical_discount
        }
    
    return {
        'game': "Prisoner's Dilemma (R=3, T=5, P=1, S=0)",
        'critical_discount': 0.5,  # (5-3)/(5-1) = 0.5
        'results_by_delta': results,
        'interpretation': (
            "Cooperation sustainable when δ ≥ 0.5. "
            "More patient players (higher δ) can sustain cooperation."
        )
    }


def example_job_market_signaling():
    """
    Spence job market signaling example.
    """
    result = signaling_game(
        sender_types=['high', 'low'],
        type_priors={'high': 0.3, 'low': 0.7},
        signals=['educate', 'no_educate'],
        signal_costs={
            'high': {'educate': 40000, 'no_educate': 0},
            'low': {'educate': 80000, 'no_educate': 0}
        },
        receiver_actions=['hire_high', 'hire_low'],
        payoffs={'high_wage': 100000, 'low_wage': 50000}
    )
    
    return {
        'model': result,
        'interpretation': (
            "High types can profitably signal through education because "
            "their cost (40k) is less than wage gain (50k). "
            "Low types cannot profitably mimic because their cost (80k) "
            "exceeds the wage gain."
        )
    }


# Export all
__all__ = [
    # Data classes
    'RepeatedGameResult',
    'SubgamePerfectResult', 
    'BayesianGameResult',
    'MechanismResult',
    
    # Repeated games
    'analyze_repeated_game',
    'folk_theorem_payoffs',
    'repeated_cournot',
    
    # Bayesian games
    'bayesian_game_2x2',
    'signaling_game',
    
    # SPE
    'backward_induction',
    'centipede_game_analysis',
    
    # Mechanism design
    'revelation_principle_example',
    'vickrey_clarke_groves',
    'myerson_optimal_auction',
    
    # Examples
    'example_repeated_prisoners_dilemma',
    'example_job_market_signaling',
]
