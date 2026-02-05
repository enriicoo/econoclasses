"""
Game Theory Examples
====================

Demonstrates the game theory module:
- Classic games (Prisoner's Dilemma, Battle of Sexes)
- Nash equilibrium finding (pure and mixed)
- Oligopoly models (Cournot, Bertrand, Stackelberg)
- Extensive form games and backward induction
- Visualization

Run: python -m econoclasses.examples.game_theory
"""

import numpy as np
import matplotlib.pyplot as plt


def example_classic_games():
    """Demonstrate classic 2x2 games and Nash equilibrium."""
    from econoclasses.game_theory import (
        prisoners_dilemma, battle_of_sexes, matching_pennies,
        stag_hunt, chicken, find_pure_nash, find_mixed_nash_2player
    )
    
    print("=" * 60)
    print("  CLASSIC 2x2 GAMES")
    print("=" * 60)
    
    # Prisoner's Dilemma
    print("\n1. PRISONER'S DILEMMA")
    print("-" * 40)
    pd = prisoners_dilemma()
    print(pd.payoff_table())
    
    pure_eq = find_pure_nash(pd)
    print(f"\nPure Nash Equilibria: {pure_eq}")
    print("→ Both defect is the unique Nash equilibrium (dominant strategy)")
    
    # Battle of the Sexes
    print("\n\n2. BATTLE OF THE SEXES")
    print("-" * 40)
    bos = battle_of_sexes()
    print(bos.payoff_table())
    
    pure_eq = find_pure_nash(bos)
    mixed_eq = find_mixed_nash_2player(bos)
    print(f"\nPure Nash Equilibria: {pure_eq}")
    print(f"Mixed Nash Equilibria: {mixed_eq}")
    print("→ Two pure equilibria (coordination) + one mixed equilibrium")
    
    # Matching Pennies
    print("\n\n3. MATCHING PENNIES (Zero-Sum)")
    print("-" * 40)
    mp = matching_pennies()
    print(mp.payoff_table())
    
    pure_eq = find_pure_nash(mp)
    mixed_eq = find_mixed_nash_2player(mp)
    print(f"\nPure Nash Equilibria: {pure_eq}")
    print(f"Mixed Nash Equilibria: {mixed_eq}")
    print("→ No pure equilibrium; unique mixed equilibrium at (0.5, 0.5)")
    
    # Stag Hunt
    print("\n\n4. STAG HUNT")
    print("-" * 40)
    sh = stag_hunt()
    print(sh.payoff_table())
    
    pure_eq = find_pure_nash(sh)
    print(f"\nPure Nash Equilibria: {pure_eq}")
    print("→ Two pure equilibria: (Stag, Stag) is Pareto-optimal")
    print("   (Hare, Hare) is risk-dominant")


def example_dominant_strategies():
    """Demonstrate dominant strategy analysis."""
    from econoclasses.game_theory import (
        prisoners_dilemma, find_dominant_strategy_equilibrium,
        iterated_elimination, NormalFormGame
    )
    
    print("\n\n" + "=" * 60)
    print("  DOMINANT STRATEGIES & ITERATED ELIMINATION")
    print("=" * 60)
    
    # Prisoner's Dilemma has dominant strategy equilibrium
    print("\n1. PRISONER'S DILEMMA - Dominant Strategy")
    print("-" * 40)
    pd = prisoners_dilemma()
    dse = find_dominant_strategy_equilibrium(pd)
    print(f"Dominant Strategy Equilibrium: {dse}")
    
    # Example with iterated elimination
    print("\n\n2. ITERATED ELIMINATION OF DOMINATED STRATEGIES")
    print("-" * 40)
    
    # Create a game that requires IESDS
    game = NormalFormGame.from_payoff_dict(
        ['Row', 'Column'],
        [['T', 'M', 'B'], ['L', 'C', 'R']],
        {
            ('T', 'L'): (3, 1), ('T', 'C'): (0, 1), ('T', 'R'): (0, 0),
            ('M', 'L'): (1, 1), ('M', 'C'): (1, 1), ('M', 'R'): (5, 0),
            ('B', 'L'): (0, 1), ('B', 'C'): (4, 1), ('B', 'R'): (0, 0),
        }
    )
    print("Original game:")
    print(game.payoff_table())
    
    reduced, log = iterated_elimination(game, strict=True, verbose=True)
    print(f"\nReduced game after IESDS:")
    print(reduced.payoff_table())


def example_oligopoly():
    """Demonstrate Cournot, Bertrand, and Stackelberg models."""
    from econoclasses.game_theory import (
        cournot_duopoly, bertrand_duopoly, stackelberg_duopoly,
        cournot_n_firms, compare_oligopoly_models
    )
    
    print("\n\n" + "=" * 60)
    print("  OLIGOPOLY MODELS")
    print("=" * 60)
    
    # Parameters
    a, b, c = 100, 1, 10  # Demand: P = 100 - Q, MC = 10
    
    # Cournot Duopoly
    print("\n1. COURNOT DUOPOLY (Quantity Competition)")
    print("-" * 40)
    cournot = cournot_duopoly(a=a, b=b, c1=c, c2=c)
    print(f"Quantities: q₁ = {cournot.q1:.2f}, q₂ = {cournot.q2:.2f}")
    print(f"Total Q = {cournot.Q:.2f}")
    print(f"Price P = {cournot.P:.2f}")
    print(f"Profits: π₁ = {cournot.profit1:.2f}, π₂ = {cournot.profit2:.2f}")
    print(f"Consumer Surplus = {cournot.consumer_surplus:.2f}")
    
    # Bertrand Duopoly
    print("\n\n2. BERTRAND DUOPOLY (Price Competition)")
    print("-" * 40)
    bertrand = bertrand_duopoly(a=a, b=b, c1=c, c2=c)
    print(f"Prices: p₁ = {bertrand.p1:.2f}, p₂ = {bertrand.p2:.2f}")
    print(f"Quantities: q₁ = {bertrand.q1:.2f}, q₂ = {bertrand.q2:.2f}")
    print(f"Profits: π₁ = {bertrand.profit1:.2f}, π₂ = {bertrand.profit2:.2f}")
    print("→ Bertrand paradox: With homogeneous products, P = MC!")
    
    # Stackelberg Duopoly
    print("\n\n3. STACKELBERG DUOPOLY (Sequential)")
    print("-" * 40)
    stackelberg = stackelberg_duopoly(a=a, b=b, c_leader=c, c_follower=c)
    print(f"Leader quantity: q_L = {stackelberg.q_leader:.2f}")
    print(f"Follower quantity: q_F = {stackelberg.q_follower:.2f}")
    print(f"Price P = {stackelberg.P:.2f}")
    print(f"Profits: π_L = {stackelberg.profit_leader:.2f}, π_F = {stackelberg.profit_follower:.2f}")
    print("→ First-mover advantage: Leader commits to higher quantity")
    
    # N-firm Cournot
    print("\n\n4. N-FIRM COURNOT")
    print("-" * 40)
    for n in [2, 3, 5, 10]:
        result = cournot_n_firms(a=a, b=b, n=n)
        print(f"n={n}: Q={result['total_quantity']:.1f}, P={result['price']:.1f}, "
              f"Total profit={sum(result['profits']):.1f}")
    print("→ As n → ∞, approaches competitive equilibrium (P → MC)")
    
    # Comparison
    print("\n\n5. MODEL COMPARISON")
    print("-" * 40)
    comparison = compare_oligopoly_models(a, b, c)
    print(f"{'Model':<15} {'Price':>10} {'Quantity':>10} {'Profit':>10}")
    print("-" * 45)
    print(f"{'Competitive':<15} {comparison['competitive']['price']:>10.2f} "
          f"{comparison['competitive']['quantity']:>10.2f} {0:>10.2f}")
    print(f"{'Bertrand':<15} {comparison['bertrand']['prices'][0]:>10.2f} "
          f"{comparison['bertrand']['total_Q']:>10.2f} {comparison['bertrand']['total_profit']:>10.2f}")
    print(f"{'Cournot':<15} {comparison['cournot']['price']:>10.2f} "
          f"{comparison['cournot']['total_Q']:>10.2f} {comparison['cournot']['total_profit']:>10.2f}")
    print(f"{'Stackelberg':<15} {comparison['stackelberg']['price']:>10.2f} "
          f"{comparison['stackelberg']['total_Q']:>10.2f} {comparison['stackelberg']['total_profit']:>10.2f}")
    print(f"{'Monopoly':<15} {comparison['monopoly']['price']:>10.2f} "
          f"{comparison['monopoly']['quantity']:>10.2f} {comparison['monopoly']['profit']:>10.2f}")


def example_extensive_form():
    """Demonstrate extensive form games and backward induction."""
    from econoclasses.game_theory import entry_game, ultimatum_game, centipede_game
    
    print("\n\n" + "=" * 60)
    print("  EXTENSIVE FORM GAMES")
    print("=" * 60)
    
    # Entry Game
    print("\n1. MARKET ENTRY GAME")
    print("-" * 40)
    game = entry_game()
    print(f"Players: {game.player_names}")
    print("\nGame tree structure:")
    print("  Entrant → Enter → Incumbent → Fight: (-1, -1)")
    print("                              → Accommodate: (2, 1)")
    print("         → Out → (0, 4)")
    
    solution = game.backward_induction()
    print(f"\nBackward induction solution: {solution}")
    print("→ Entrant enters, Incumbent accommodates (credible threat analysis)")
    
    # Ultimatum Game
    print("\n\n2. ULTIMATUM GAME")
    print("-" * 40)
    ug = ultimatum_game(pie=10, offers=[0, 5, 10])
    print(f"Splitting ${10}")
    solution = ug.backward_induction()
    print(f"Backward induction: {solution}")
    print("→ Subgame perfect: Proposer offers $0, Responder accepts any offer ≥ $0")
    print("   (In experiments, people often reject 'unfair' offers!)")
    
    # Centipede Game
    print("\n\n3. CENTIPEDE GAME")
    print("-" * 40)
    cg = centipede_game(n_rounds=4)
    solution = cg.backward_induction()
    print(f"Backward induction: {solution}")
    print("→ Backward induction predicts Take at first node")
    print("   (But in experiments, players often Pass several rounds)")


def example_mixed_strategies():
    """Demonstrate mixed strategy equilibrium computation."""
    from econoclasses.game_theory import (
        matching_pennies, battle_of_sexes, chicken,
        find_mixed_nash_2player, maxmin_strategy, solve_zero_sum
    )
    
    print("\n\n" + "=" * 60)
    print("  MIXED STRATEGIES")
    print("=" * 60)
    
    # Matching Pennies - pure zero-sum
    print("\n1. MATCHING PENNIES (Zero-Sum)")
    print("-" * 40)
    mp = matching_pennies()
    A = mp.payoff_arrays[0]
    
    solution = solve_zero_sum(A)
    print(f"Row player strategy: {solution['row_strategy']}")
    print(f"Column player strategy: {solution['col_strategy']}")
    print(f"Game value: {solution['value']:.4f}")
    print("→ Both mix 50-50; value = 0 (fair game)")
    
    # Battle of the Sexes
    print("\n\n2. BATTLE OF THE SEXES")
    print("-" * 40)
    bos = battle_of_sexes(his_pref=3, her_pref=2, mismatch=0)
    mixed_eq = find_mixed_nash_2player(bos)
    
    # Find the interior mixed equilibrium
    for eq in mixed_eq:
        if 0.01 < eq.probabilities[0][0] < 0.99:
            print(f"Mixed equilibrium: {eq}")
            print(f"Player 1 plays Opera with prob {eq.probabilities[0][0]:.4f}")
            print(f"Player 2 plays Opera with prob {eq.probabilities[1][0]:.4f}")
    
    # Security strategies
    print("\n\n3. SECURITY (MAXMIN) STRATEGIES")
    print("-" * 40)
    game = chicken()
    for player in [0, 1]:
        strat, value = maxmin_strategy(game, player)
        print(f"Player {player+1} maxmin strategy: {strat}")
        print(f"Player {player+1} security value: {value:.4f}")


def example_plotting():
    """Create visualizations for game theory concepts."""
    from econoclasses.game_theory import (
        prisoners_dilemma, battle_of_sexes, matching_pennies,
        cournot_duopoly, entry_game
    )
    from econoclasses.plotting import (
        plot_payoff_matrix, plot_best_response,
        plot_cournot_best_response, plot_oligopoly_comparison,
        plot_game_tree
    )
    
    print("\n\n" + "=" * 60)
    print("  GENERATING PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Payoff matrix
    pd = prisoners_dilemma()
    plot_payoff_matrix(pd, ax=axes[0, 0], title="Prisoner's Dilemma")
    
    # 2. Best response for Battle of Sexes
    bos = battle_of_sexes()
    plot_best_response(bos, ax=axes[0, 1])
    
    # 3. Cournot best response
    plot_cournot_best_response(a=100, b=1, c1=10, c2=10, ax=axes[1, 0])
    
    # 4. Entry game tree
    game = entry_game()
    solution = game.backward_induction()
    plot_game_tree(game, ax=axes[1, 1], solution=solution)
    
    plt.tight_layout()
    plt.savefig('/tmp/game_theory_plots.png', dpi=150, bbox_inches='tight')
    print("Saved plots to /tmp/game_theory_plots.png")
    
    # Oligopoly comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    plot_oligopoly_comparison(a=100, b=1, c=10, ax=axes2[0])
    # The function returns all axes, so we need to handle this differently
    plt.figure(figsize=(14, 5))
    plot_oligopoly_comparison(a=100, b=1, c=10)
    plt.savefig('/tmp/oligopoly_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved oligopoly comparison to /tmp/oligopoly_comparison.png")
    
    plt.close('all')


def run_all_examples():
    """Run all game theory examples."""
    example_classic_games()
    example_dominant_strategies()
    example_oligopoly()
    example_extensive_form()
    example_mixed_strategies()
    
    try:
        example_plotting()
    except Exception as e:
        print(f"\nPlotting skipped: {e}")


if __name__ == "__main__":
    run_all_examples()
