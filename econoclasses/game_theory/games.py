"""
Game theory foundations: Normal form and Extensive form games.

Provides base classes for representing strategic interactions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from itertools import product


@dataclass
class Strategy:
    """A strategy for a player."""
    name: str
    index: int
    
    def __repr__(self):
        return f"Strategy('{self.name}')"


@dataclass
class Player:
    """A player in the game."""
    name: str
    strategies: List[Strategy]
    index: int = 0
    
    @property
    def n_strategies(self) -> int:
        return len(self.strategies)
    
    def strategy(self, name_or_index: Union[str, int]) -> Strategy:
        """Get strategy by name or index."""
        if isinstance(name_or_index, int):
            return self.strategies[name_or_index]
        for s in self.strategies:
            if s.name == name_or_index:
                return s
        raise ValueError(f"Strategy '{name_or_index}' not found")
    
    def __repr__(self):
        strat_names = [s.name for s in self.strategies]
        return f"Player('{self.name}', strategies={strat_names})"


@dataclass
class StrategyProfile:
    """A combination of strategies, one for each player."""
    strategies: Tuple[Strategy, ...]
    
    def __repr__(self):
        names = tuple(s.name for s in self.strategies)
        return f"StrategyProfile{names}"


@dataclass
class Outcome:
    """Outcome of a strategy profile."""
    profile: StrategyProfile
    payoffs: Tuple[float, ...]  # One payoff per player
    
    def __repr__(self):
        return f"Outcome({self.profile}, payoffs={self.payoffs})"


class NormalFormGame:
    """
    A game in normal (strategic) form.
    
    Defined by:
    - Set of players
    - Strategy set for each player
    - Payoff function mapping strategy profiles to payoffs
    
    Example
    -------
    >>> # Prisoner's Dilemma
    >>> payoffs = {
    ...     ('C', 'C'): (-1, -1),
    ...     ('C', 'D'): (-3, 0),
    ...     ('D', 'C'): (0, -3),
    ...     ('D', 'D'): (-2, -2)
    ... }
    >>> game = NormalFormGame.from_payoff_dict(
    ...     ['Player 1', 'Player 2'],
    ...     [['C', 'D'], ['C', 'D']],
    ...     payoffs
    ... )
    >>> print(game.payoff_matrix(0))  # Player 1's payoffs
    """
    
    def __init__(self, players: List[Player], payoff_arrays: List[np.ndarray]):
        """
        Initialize game with players and payoff arrays.
        
        Parameters
        ----------
        players : list of Player
            The players in the game
        payoff_arrays : list of np.ndarray
            One array per player, shape = (n_strats_1, n_strats_2, ...)
            Entry [i, j, ...] is player's payoff when player 1 plays i, player 2 plays j, etc.
        """
        self.players = players
        self.n_players = len(players)
        self.payoff_arrays = payoff_arrays
        
        # Validate dimensions
        expected_shape = tuple(p.n_strategies for p in players)
        for i, arr in enumerate(payoff_arrays):
            if arr.shape != expected_shape:
                raise ValueError(f"Payoff array {i} has wrong shape: {arr.shape}, expected {expected_shape}")
    
    @classmethod
    def from_payoff_dict(
        cls,
        player_names: List[str],
        strategy_names: List[List[str]],
        payoffs: Dict[Tuple[str, ...], Tuple[float, ...]]
    ) -> 'NormalFormGame':
        """
        Create game from dictionary of payoffs.
        
        Parameters
        ----------
        player_names : list of str
            Names of the players
        strategy_names : list of list of str
            Strategy names for each player
        payoffs : dict
            Maps (strategy_name_1, strategy_name_2, ...) -> (payoff_1, payoff_2, ...)
        
        Example
        -------
        >>> payoffs = {
        ...     ('C', 'C'): (-1, -1),
        ...     ('C', 'D'): (-3, 0),
        ...     ('D', 'C'): (0, -3),
        ...     ('D', 'D'): (-2, -2)
        ... }
        >>> game = NormalFormGame.from_payoff_dict(
        ...     ['P1', 'P2'], [['C', 'D'], ['C', 'D']], payoffs
        ... )
        """
        n_players = len(player_names)
        
        # Create players
        players = []
        for i, (name, strats) in enumerate(zip(player_names, strategy_names)):
            strategies = [Strategy(s, j) for j, s in enumerate(strats)]
            players.append(Player(name, strategies, i))
        
        # Create payoff arrays
        shape = tuple(len(s) for s in strategy_names)
        payoff_arrays = [np.zeros(shape) for _ in range(n_players)]
        
        # Fill in payoffs
        for strat_combo, payoff_tuple in payoffs.items():
            # Convert strategy names to indices
            indices = tuple(
                strategy_names[i].index(strat_combo[i])
                for i in range(n_players)
            )
            for player_idx, payoff in enumerate(payoff_tuple):
                payoff_arrays[player_idx][indices] = payoff
        
        return cls(players, payoff_arrays)
    
    @classmethod
    def from_bimatrix(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        player_names: List[str] = None,
        row_strategies: List[str] = None,
        col_strategies: List[str] = None
    ) -> 'NormalFormGame':
        """
        Create 2-player game from bimatrix representation.
        
        Parameters
        ----------
        A : np.ndarray
            Payoff matrix for row player (player 1)
        B : np.ndarray
            Payoff matrix for column player (player 2)
        """
        if A.shape != B.shape:
            raise ValueError("Payoff matrices must have same shape")
        
        n_rows, n_cols = A.shape
        
        if player_names is None:
            player_names = ['Row Player', 'Column Player']
        if row_strategies is None:
            row_strategies = [f'R{i+1}' for i in range(n_rows)]
        if col_strategies is None:
            col_strategies = [f'C{i+1}' for i in range(n_cols)]
        
        players = [
            Player(player_names[0], [Strategy(s, i) for i, s in enumerate(row_strategies)], 0),
            Player(player_names[1], [Strategy(s, i) for i, s in enumerate(col_strategies)], 1)
        ]
        
        return cls(players, [A, B])
    
    # =========================================================================
    # ACCESSORS
    # =========================================================================
    
    def payoff(self, strategy_profile: Union[Tuple, StrategyProfile], player: int) -> float:
        """Get payoff for a player given a strategy profile."""
        if isinstance(strategy_profile, StrategyProfile):
            indices = tuple(s.index for s in strategy_profile.strategies)
        else:
            # Assume tuple of strategy names or indices
            indices = []
            for i, s in enumerate(strategy_profile):
                if isinstance(s, str):
                    indices.append(self.players[i].strategy(s).index)
                elif isinstance(s, Strategy):
                    indices.append(s.index)
                else:
                    indices.append(s)
            indices = tuple(indices)
        
        return self.payoff_arrays[player][indices]
    
    def all_payoffs(self, strategy_profile: Union[Tuple, StrategyProfile]) -> Tuple[float, ...]:
        """Get payoffs for all players."""
        return tuple(self.payoff(strategy_profile, i) for i in range(self.n_players))
    
    def payoff_matrix(self, player: int) -> np.ndarray:
        """Get the full payoff matrix for a player."""
        return self.payoff_arrays[player]
    
    def strategy_profiles(self) -> List[StrategyProfile]:
        """Generate all possible strategy profiles."""
        strat_lists = [p.strategies for p in self.players]
        return [StrategyProfile(combo) for combo in product(*strat_lists)]
    
    # =========================================================================
    # BEST RESPONSE
    # =========================================================================
    
    def best_response(self, player: int, other_strategies: Dict[int, int]) -> List[int]:
        """
        Find best response(s) for a player given others' strategies.
        
        Parameters
        ----------
        player : int
            Index of the player finding best response
        other_strategies : dict
            Maps player index -> strategy index for all other players
        
        Returns
        -------
        list of int
            Indices of best response strategies (may have ties)
        """
        best_payoff = float('-inf')
        best_responses = []
        
        for strat_idx in range(self.players[player].n_strategies):
            # Build the full strategy profile
            profile = []
            for i in range(self.n_players):
                if i == player:
                    profile.append(strat_idx)
                else:
                    profile.append(other_strategies[i])
            
            payoff = self.payoff(tuple(profile), player)
            
            if payoff > best_payoff:
                best_payoff = payoff
                best_responses = [strat_idx]
            elif payoff == best_payoff:
                best_responses.append(strat_idx)
        
        return best_responses
    
    def best_response_to_mixed(self, player: int, 
                                other_mixed: List[np.ndarray]) -> List[int]:
        """
        Find best response(s) to mixed strategies of other players.
        
        For 2-player games.
        """
        if self.n_players != 2:
            raise NotImplementedError("Only implemented for 2-player games")
        
        other = 1 - player
        other_probs = other_mixed[other]
        
        expected_payoffs = []
        for my_strat in range(self.players[player].n_strategies):
            exp_payoff = 0
            for their_strat, prob in enumerate(other_probs):
                if player == 0:
                    profile = (my_strat, their_strat)
                else:
                    profile = (their_strat, my_strat)
                exp_payoff += prob * self.payoff(profile, player)
            expected_payoffs.append(exp_payoff)
        
        max_payoff = max(expected_payoffs)
        return [i for i, p in enumerate(expected_payoffs) if abs(p - max_payoff) < 1e-10]
    
    # =========================================================================
    # DOMINANCE
    # =========================================================================
    
    def is_strictly_dominated(self, player: int, strategy: int) -> Optional[int]:
        """
        Check if strategy is strictly dominated.
        
        Returns index of dominating strategy, or None.
        """
        for other_strat in range(self.players[player].n_strategies):
            if other_strat == strategy:
                continue
            
            # Check if other_strat strictly dominates strategy
            dominates = True
            for profile_indices in product(*[range(p.n_strategies) for p in self.players]):
                if profile_indices[player] == strategy:
                    # Compare payoffs
                    profile_with_other = tuple(
                        other_strat if i == player else profile_indices[i]
                        for i in range(self.n_players)
                    )
                    if self.payoff(profile_with_other, player) <= self.payoff(profile_indices, player):
                        dominates = False
                        break
            
            if dominates:
                return other_strat
        
        return None
    
    def is_weakly_dominated(self, player: int, strategy: int) -> Optional[int]:
        """
        Check if strategy is weakly dominated.
        
        Returns index of dominating strategy, or None.
        """
        for other_strat in range(self.players[player].n_strategies):
            if other_strat == strategy:
                continue
            
            strictly_better_somewhere = False
            at_least_as_good = True
            
            for profile_indices in product(*[range(p.n_strategies) for p in self.players]):
                if profile_indices[player] == strategy:
                    profile_with_other = tuple(
                        other_strat if i == player else profile_indices[i]
                        for i in range(self.n_players)
                    )
                    payoff_other = self.payoff(profile_with_other, player)
                    payoff_this = self.payoff(profile_indices, player)
                    
                    if payoff_other < payoff_this:
                        at_least_as_good = False
                        break
                    if payoff_other > payoff_this:
                        strictly_better_somewhere = True
            
            if at_least_as_good and strictly_better_somewhere:
                return other_strat
        
        return None
    
    def dominant_strategy(self, player: int) -> Optional[int]:
        """
        Find strictly dominant strategy for player, if one exists.
        """
        for strat in range(self.players[player].n_strategies):
            is_dominant = True
            for other_strat in range(self.players[player].n_strategies):
                if other_strat == strat:
                    continue
                if self.is_strictly_dominated(player, other_strat) != strat:
                    is_dominant = False
                    break
            
            if is_dominant:
                return strat
        
        return None
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def __repr__(self):
        player_strs = [f"{p.name}: {[s.name for s in p.strategies]}" for p in self.players]
        return f"NormalFormGame({', '.join(player_strs)})"
    
    def payoff_table(self) -> str:
        """
        Generate ASCII payoff table (2-player games only).
        """
        if self.n_players != 2:
            return "Payoff table only available for 2-player games"
        
        p1, p2 = self.players
        col_width = 15
        
        # Header
        header = f"{'':<{col_width}}"
        for s in p2.strategies:
            header += f"{s.name:^{col_width}}"
        
        lines = [
            f"Game: {p1.name} (rows) vs {p2.name} (columns)",
            "=" * (col_width * (p2.n_strategies + 1)),
            header,
            "-" * (col_width * (p2.n_strategies + 1)),
        ]
        
        for i, s1 in enumerate(p1.strategies):
            row = f"{s1.name:<{col_width}}"
            for j, s2 in enumerate(p2.strategies):
                pay1 = self.payoff_arrays[0][i, j]
                pay2 = self.payoff_arrays[1][i, j]
                row += f"({pay1:.1f}, {pay2:.1f})".center(col_width)
            lines.append(row)
        
        return '\n'.join(lines)


@dataclass
class GameNode:
    """
    A node in an extensive form game tree.
    """
    name: str
    player: Optional[int] = None  # None for terminal/chance nodes
    children: Dict[str, 'GameNode'] = field(default_factory=dict)
    payoffs: Optional[Tuple[float, ...]] = None  # Only for terminal nodes
    info_set: Optional[str] = None  # Information set label
    
    @property
    def is_terminal(self) -> bool:
        return self.payoffs is not None
    
    @property
    def is_chance(self) -> bool:
        return self.player is None and not self.is_terminal
    
    def add_child(self, action: str, child: 'GameNode') -> 'GameNode':
        """Add a child node for an action."""
        self.children[action] = child
        return child


class ExtensiveFormGame:
    """
    A game in extensive form (game tree).
    
    Represents sequential games with:
    - Decision nodes (assigned to players)
    - Chance nodes (nature moves)
    - Terminal nodes (payoffs)
    - Information sets (for imperfect information)
    
    Example
    -------
    >>> # Entry game: Entrant decides to Enter or Stay Out
    >>> # If Enter, Incumbent decides to Fight or Accommodate
    >>> root = GameNode("root", player=0)
    >>> out = root.add_child("Out", GameNode("out", payoffs=(0, 2)))
    >>> enter = root.add_child("Enter", GameNode("enter", player=1))
    >>> enter.add_child("Fight", GameNode("fight", payoffs=(-1, -1)))
    >>> enter.add_child("Accommodate", GameNode("acc", payoffs=(1, 1)))
    >>> game = ExtensiveFormGame(root, ["Entrant", "Incumbent"])
    """
    
    def __init__(self, root: GameNode, player_names: List[str]):
        self.root = root
        self.player_names = player_names
        self.n_players = len(player_names)
    
    def terminal_nodes(self) -> List[GameNode]:
        """Get all terminal nodes."""
        terminals = []
        
        def collect(node):
            if node.is_terminal:
                terminals.append(node)
            for child in node.children.values():
                collect(child)
        
        collect(self.root)
        return terminals
    
    def backward_induction(self) -> Dict[str, str]:
        """
        Solve game by backward induction (subgame perfect equilibrium).
        
        Returns dict mapping node names to optimal actions.
        Only works for perfect information games.
        """
        solution = {}
        
        def solve_node(node: GameNode) -> Tuple[float, ...]:
            if node.is_terminal:
                return node.payoffs
            
            # Solve all children first
            child_values = {}
            for action, child in node.children.items():
                child_values[action] = solve_node(child)
            
            # Find best action for current player
            player = node.player
            best_action = max(child_values.keys(), 
                            key=lambda a: child_values[a][player])
            solution[node.name] = best_action
            
            return child_values[best_action]
        
        solve_node(self.root)
        return solution
    
    def to_normal_form(self) -> NormalFormGame:
        """
        Convert to normal form game.
        
        Warning: exponential in game tree size.
        """
        # For 2-player games, enumerate all strategies
        if self.n_players != 2:
            raise NotImplementedError("Only 2-player conversion supported")
        
        # Find all decision nodes for each player
        player_nodes = [[], []]
        
        def find_nodes(node):
            if node.player is not None and not node.is_terminal:
                player_nodes[node.player].append(node)
            for child in node.children.values():
                find_nodes(child)
        
        find_nodes(self.root)
        
        # Generate all pure strategies (mapping from nodes to actions)
        from itertools import product as iter_product
        
        def enumerate_strategies(nodes):
            if not nodes:
                return [{}]
            action_choices = [list(n.children.keys()) for n in nodes]
            strategies = []
            for combo in iter_product(*action_choices):
                strat = {nodes[i].name: combo[i] for i in range(len(nodes))}
                strategies.append(strat)
            return strategies
        
        strategies = [enumerate_strategies(player_nodes[i]) for i in range(2)]
        
        # Build payoff matrices
        n1, n2 = len(strategies[0]), len(strategies[1])
        A = np.zeros((n1, n2))
        B = np.zeros((n1, n2))
        
        def evaluate(strat0, strat1) -> Tuple[float, ...]:
            """Follow strategies through tree to terminal node."""
            combined = {**strat0, **strat1}
            node = self.root
            while not node.is_terminal:
                action = combined.get(node.name)
                if action is None:
                    # Chance node or error
                    action = list(node.children.keys())[0]
                node = node.children[action]
            return node.payoffs
        
        for i, s1 in enumerate(strategies[0]):
            for j, s2 in enumerate(strategies[1]):
                payoffs = evaluate(s1, s2)
                A[i, j] = payoffs[0]
                B[i, j] = payoffs[1]
        
        # Create strategy names
        def strat_name(strat_dict):
            if not strat_dict:
                return "∅"
            return "-".join(f"{v}" for v in strat_dict.values())
        
        row_strats = [strat_name(s) for s in strategies[0]]
        col_strats = [strat_name(s) for s in strategies[1]]
        
        return NormalFormGame.from_bimatrix(
            A, B, 
            self.player_names,
            row_strats, col_strats
        )
    
    def __repr__(self):
        return f"ExtensiveFormGame(players={self.player_names})"
