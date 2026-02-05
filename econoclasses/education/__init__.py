"""
Educational tools: step-by-step derivations and explanations.
"""

import sympy as sp
from typing import List
from dataclasses import dataclass

from ..core import X, Y, K, L, Px, Py, I, w, r, lam


@dataclass
class DerivationStep:
    number: int
    title: str
    explanation: str
    latex: str
    
    def to_markdown(self) -> str:
        return f"**Step {self.number}: {self.title}**\n\n{self.explanation}\n\n$${self.latex}$$\n"


class Derivation:
    def __init__(self, title: str, steps: List[DerivationStep]):
        self.title = title
        self.steps = steps
    
    def to_markdown(self) -> str:
        lines = [f"# {self.title}\n"]
        for step in self.steps:
            lines.append(step.to_markdown())
        return '\n'.join(lines)
    
    def to_latex(self) -> str:
        lines = [f"\\section{{{self.title}}}\n"]
        for step in self.steps:
            lines.append(f"\\subsection*{{Step {step.number}: {step.title}}}")
            lines.append(step.explanation)
            lines.append(f"$${step.latex}$$\n")
        return '\n'.join(lines)
    
    def print(self):
        print(f"\n{'='*60}\n  {self.title}\n{'='*60}")
        for step in self.steps:
            print(f"\nStep {step.number}: {step.title}")
            print(f"  {step.explanation}")


# =============================================================================
# CONSUMER THEORY DERIVATIONS
# =============================================================================

def derive_marshallian_demand(utility) -> Derivation:
    """Generate step-by-step Marshallian demand derivation."""
    U_expr = utility.expr
    steps = []
    
    steps.append(DerivationStep(
        1, "The Optimization Problem",
        "Maximize utility subject to budget constraint:",
        f"\\max_{{X,Y}} {sp.latex(U_expr)} \\quad \\text{{s.t.}} \\quad P_x X + P_y Y = I"
    ))
    
    L_expr = U_expr - lam * (Px * X + Py * Y - I)
    steps.append(DerivationStep(
        2, "Form the Lagrangian",
        "Combine objective and constraint:",
        f"\\mathcal{{L}} = {sp.latex(U_expr)} - \\lambda(P_x X + P_y Y - I)"
    ))
    
    dL_dX = sp.diff(L_expr, X)
    dL_dY = sp.diff(L_expr, Y)
    
    steps.append(DerivationStep(
        3, "First-Order Conditions",
        "Take partial derivatives and set to zero:",
        f"\\frac{{\\partial \\mathcal{{L}}}}{{\\partial X}} = {sp.latex(dL_dX)} = 0"
    ))
    
    MRS = sp.simplify(sp.diff(U_expr, X) / sp.diff(U_expr, Y))
    steps.append(DerivationStep(
        4, "Tangency Condition",
        "MRS equals price ratio (slope of IC = slope of budget):",
        f"MRS = {sp.latex(MRS)} = \\frac{{P_x}}{{P_y}}"
    ))
    
    x_num, y_num = utility.marshallian_demand
    u_num = utility.utility_at(x_num, y_num)
    steps.append(DerivationStep(
        5, "Numerical Solution",
        f"At Px={utility.price_x}, Py={utility.price_y}, I={utility.income}:",
        f"X^* = {x_num:.4f}, \\quad Y^* = {y_num:.4f}, \\quad U^* = {u_num:.4f}"
    ))
    
    return Derivation(f"Marshallian Demand: {utility.form_name}", steps)


def derive_slutsky_equation(utility, new_price_x: float) -> Derivation:
    """Generate step-by-step Slutsky decomposition."""
    decomp = utility.slutsky_decomposition(new_price_x=new_price_x, good='X')
    steps = []
    
    steps.append(DerivationStep(
        1, "Price Change",
        f"Price of X changes from {utility.price_x} to {new_price_x}:",
        f"\\Delta P_x = {new_price_x - utility.price_x:+.2f}"
    ))
    
    steps.append(DerivationStep(
        2, "Original Bundle (Point A)",
        "Optimal bundle before price change:",
        f"A = ({decomp['original_bundle'][0]:.3f}, {decomp['original_bundle'][1]:.3f})"
    ))
    
    steps.append(DerivationStep(
        3, "Compensated Bundle (Point B)",
        "Bundle on original IC at new prices (Hicksian demand):",
        f"B = ({decomp['compensated_bundle'][0]:.3f}, {decomp['compensated_bundle'][1]:.3f})"
    ))
    
    steps.append(DerivationStep(
        4, "Final Bundle (Point C)",
        "Optimal bundle at new prices with original income:",
        f"C = ({decomp['new_bundle'][0]:.3f}, {decomp['new_bundle'][1]:.3f})"
    ))
    
    steps.append(DerivationStep(
        5, "Decomposition",
        "A→B is substitution effect, B→C is income effect:",
        f"\\Delta X^{{SE}} = {decomp['substitution_effect']:+.4f}, \\quad "
        f"\\Delta X^{{IE}} = {decomp['income_effect']:+.4f}, \\quad "
        f"\\Delta X^{{total}} = {decomp['total_effect']:+.4f}"
    ))
    
    steps.append(DerivationStep(
        6, "Slutsky Equation",
        "Total effect = Substitution effect + Income effect:",
        f"\\frac{{\\partial x}}{{\\partial p_x}} = "
        f"\\frac{{\\partial h}}{{\\partial p_x}} - x \\frac{{\\partial x}}{{\\partial I}}"
    ))
    
    return Derivation("Slutsky Decomposition", steps)


# =============================================================================
# PRODUCTION THEORY DERIVATIONS
# =============================================================================

def derive_cost_minimization(tech, Q_target: float, wage: float, rental: float) -> Derivation:
    """Generate step-by-step cost minimization derivation."""
    Q_expr = tech.expr
    steps = []
    
    steps.append(DerivationStep(
        1, "The Optimization Problem",
        f"Minimize cost subject to producing Q = {Q_target}:",
        f"\\min_{{K,L}} rK + wL \\quad \\text{{s.t.}} \\quad {sp.latex(Q_expr)} = {Q_target}"
    ))
    
    steps.append(DerivationStep(
        2, "Form the Lagrangian",
        "Combine objective and constraint:",
        f"\\mathcal{{L}} = rK + wL - \\mu({sp.latex(Q_expr)} - {Q_target})"
    ))
    
    MPK = tech.marginal_product_K
    MPL = tech.marginal_product_L
    steps.append(DerivationStep(
        3, "First-Order Conditions",
        "Take derivatives with respect to K and L:",
        f"r = \\mu \\cdot MP_K = \\mu \\cdot {sp.latex(MPK)}, \\quad "
        f"w = \\mu \\cdot MP_L = \\mu \\cdot {sp.latex(MPL)}"
    ))
    
    MRTS = sp.simplify(MPL / MPK)
    steps.append(DerivationStep(
        4, "Tangency Condition",
        "MRTS equals factor price ratio (isoquant tangent to isocost):",
        f"MRTS = \\frac{{MP_L}}{{MP_K}} = {sp.latex(MRTS)} = \\frac{{w}}{{r}} = \\frac{{{wage}}}{{{rental}}}"
    ))
    
    sol = tech.cost_minimize(Q_target, wage, rental)
    steps.append(DerivationStep(
        5, "Numerical Solution",
        f"At w={wage}, r={rental}, Q={Q_target}:",
        f"K^* = {sol.K:.4f}, \\quad L^* = {sol.L:.4f}, \\quad C^* = {sol.cost:.4f}"
    ))
    
    steps.append(DerivationStep(
        6, "Interpretation",
        "Cost-minimizing input ratio depends only on factor prices, not output level:",
        f"\\frac{{K^*}}{{L^*}} = {sol.K/sol.L:.4f}"
    ))
    
    return Derivation(f"Cost Minimization: {tech.form_name}", steps)


def derive_profit_maximization(firm, price: float) -> Derivation:
    """Generate step-by-step profit maximization derivation."""
    tech = firm.technology
    Q_expr = tech.expr
    steps = []
    
    steps.append(DerivationStep(
        1, "The Optimization Problem",
        f"Maximize profit given output price P = {price}:",
        f"\\max_{{K,L}} P \\cdot {sp.latex(Q_expr)} - rK - wL"
    ))
    
    MPK = tech.marginal_product_K
    MPL = tech.marginal_product_L
    steps.append(DerivationStep(
        2, "First-Order Conditions",
        "Value of marginal product equals factor price:",
        f"P \\cdot MP_K = r \\Rightarrow {price} \\cdot {sp.latex(MPK)} = {firm.rental}"
    ))
    
    steps.append(DerivationStep(
        3, "Labor Condition",
        "Similarly for labor:",
        f"P \\cdot MP_L = w \\Rightarrow {price} \\cdot {sp.latex(MPL)} = {firm.wage}"
    ))
    
    Q_star = firm.supply_at_price(price)
    profit = firm.profit_at_price(price)
    inputs = firm.conditional_input_demand(Q_star) if Q_star > 0 else {'K': 0, 'L': 0}
    
    steps.append(DerivationStep(
        4, "Numerical Solution",
        f"At P={price}, w={firm.wage}, r={firm.rental}:",
        f"K^* = {inputs['K']:.4f}, \\quad L^* = {inputs['L']:.4f}, \\quad Q^* = {Q_star:.4f}"
    ))
    
    steps.append(DerivationStep(
        5, "Profit",
        "Revenue minus cost:",
        f"\\pi = P \\cdot Q - rK - wL = {price} \\times {Q_star:.2f} - {profit - firm.profit_at_price(price) + profit:.2f} = {profit:.4f}"
    ))
    
    return Derivation(f"Profit Maximization: {firm.name}", steps)


# =============================================================================
# EQUILIBRIUM DERIVATIONS
# =============================================================================

def derive_exchange_equilibrium(economy) -> Derivation:
    """Generate derivation for exchange economy equilibrium."""
    steps = []
    
    steps.append(DerivationStep(
        1, "Setup",
        f"Two consumers: {economy.consumer_a.name} and {economy.consumer_b.name}",
        f"\\omega^A = ({economy.consumer_a.endowment['X']}, {economy.consumer_a.endowment['Y']}), \\quad "
        f"\\omega^B = ({economy.consumer_b.endowment['X']}, {economy.consumer_b.endowment['Y']})"
    ))
    
    steps.append(DerivationStep(
        2, "Budget Constraints",
        "Each consumer's budget equals value of endowment:",
        "P_x X^i + P_y Y^i = P_x \\omega^i_x + P_y \\omega^i_y"
    ))
    
    steps.append(DerivationStep(
        3, "Market Clearing",
        "Total demand equals total supply for each good:",
        f"X^A + X^B = {economy.total_x}, \\quad Y^A + Y^B = {economy.total_y}"
    ))
    
    steps.append(DerivationStep(
        4, "Walras' Law",
        "If one market clears, the other clears automatically:",
        "P_x Z_x + P_y Z_y = 0 \\quad \\text{(sum of excess demands = 0)}"
    ))
    
    eq = economy.find_equilibrium()
    if eq.converged:
        steps.append(DerivationStep(
            5, "Equilibrium Prices",
            "Prices that clear markets (normalizing Py = 1):",
            f"P_x/P_y = {eq.prices['X']:.4f}"
        ))
        
        steps.append(DerivationStep(
            6, "Equilibrium Allocation",
            "Final consumption bundles:",
            f"(X^A, Y^A) = ({eq.allocation.consumer_a['X']:.3f}, {eq.allocation.consumer_a['Y']:.3f}), \\quad "
            f"(X^B, Y^B) = ({eq.allocation.consumer_b['X']:.3f}, {eq.allocation.consumer_b['Y']:.3f})"
        ))
        
        steps.append(DerivationStep(
            7, "First Welfare Theorem",
            "Competitive equilibrium is Pareto efficient:",
            f"\\text{{Is Pareto efficient? }} {economy.is_pareto_efficient(eq.allocation)}"
        ))
    
    return Derivation("Exchange Economy Equilibrium", steps)


# =============================================================================
# EXPLANATIONS
# =============================================================================

def explain_utility_type(form_name: str) -> str:
    """Return educational explanation of a utility function type."""
    explanations = {
        'cobb-douglas': """
**Cobb-Douglas Utility: U = X^α Y^β**

Key Properties:
- Constant expenditure shares: spend α/(α+β) of income on X
- Unit elastic demand: price elasticity = -1
- Both goods are normal: income elasticity = 1
- Indifference curves never touch axes (both goods essential)

Economic Intuition:
The Cobb-Douglas form implies consumers always buy some of both goods,
regardless of prices. As price rises, quantity falls proportionally
so expenditure stays constant.
""",
        'perfect-substitutes': """
**Perfect Substitutes: U = αX + βY**

Key Properties:
- Constant MRS = α/β (straight-line indifference curves)
- Corner solutions: buy only the better deal
- Infinite price elasticity at the switching point
- Zero income effect

Economic Intuition:
Goods are interchangeable at rate α/β. Consumer compares "bang per buck"
(α/Px vs β/Py) and buys entirely whichever is cheaper per util.
""",
        'perfect-complements': """
**Perfect Complements: U = min(αX, βY)**

Key Properties:
- L-shaped indifference curves
- Always consume at kink: αX = βY
- Zero substitution effect (Hicksian demand vertical)
- Pure income effect

Economic Intuition:
Goods must be used together in fixed proportions (like left and right shoes).
Extra units of one good without the other provide no utility.
""",
        'ces': """
**CES (Constant Elasticity of Substitution): U = [αX^ρ + (1-α)Y^ρ]^(1/ρ)**

Key Properties:
- Elasticity of substitution: σ = 1/(1-ρ)
- Nests other forms: ρ→0 (CD), ρ=1 (linear), ρ→-∞ (Leontief)
- Flexible for empirical work

Economic Intuition:
Parameter ρ controls how easily consumers substitute between goods.
Higher ρ means goods are more substitutable.
""",
        'quasilinear': """
**Quasilinear: U = X + α ln(Y)**

Key Properties:
- Demand for Y is income-independent!
- All income effects fall on X (the numeraire)
- No wealth effects for Y
- Useful for partial equilibrium analysis

Economic Intuition:
One good (X) serves as "money" or numeraire. Consumer first optimizes
over Y, then spends remaining income on X. Simplifies analysis
when we want to ignore income effects.
""",
        'stone-geary': """
**Stone-Geary: U = (X-γx)^α (Y-γy)^β**

Key Properties:
- Subsistence levels: must consume at least γx of X, γy of Y
- Cobb-Douglas on "supernumerary" income (above subsistence)
- Engel curves are linear but don't pass through origin
- Good for modeling necessities vs luxuries

Economic Intuition:
Models basic needs that must be met before discretionary spending.
Poor consumers spend most income on subsistence; rich consumers
behave like Cobb-Douglas on their excess income.
""",
    }
    return explanations.get(form_name, f"No detailed explanation available for '{form_name}'")


def explain_production_type(form_name: str) -> str:
    """Return educational explanation of a production function type."""
    explanations = {
        'cobb-douglas': """
**Cobb-Douglas Production: Q = A K^α L^(1-α)**

Key Properties:
- Constant returns to scale (exponents sum to 1)
- Factor shares equal exponents: capital gets α of output
- Elasticity of substitution = 1
- Isoquants are smooth, convex curves

Economic Intuition:
The workhorse of production theory. Capital share α determines
income distribution between capital and labor. Widely used because
it matches many empirical regularities.
""",
        'leontief': """
**Leontief (Fixed Proportions): Q = A × min(αK, βL)**

Key Properties:
- L-shaped isoquants
- Zero substitution between inputs
- Input ratio fixed at β/α regardless of prices
- No input substitution, only scale effects

Economic Intuition:
Technology requires inputs in exact proportions (like recipes).
One machine needs exactly 2 workers; extra workers without machines
are useless. Common in short-run production.
""",
        'ces': """
**CES Production: Q = A [αK^ρ + (1-α)L^ρ]^(1/ρ)**

Key Properties:
- Elasticity of substitution: σ = 1/(1-ρ)
- Nests CD (ρ→0), linear (ρ=1), Leontief (ρ→-∞)
- Allows testing substitutability empirically

Economic Intuition:
Flexible form that allows data to determine how substitutable
inputs are. Crucial for analyzing automation, minimum wage effects,
and factor income shares.
""",
    }
    return explanations.get(form_name, f"No detailed explanation available for '{form_name}'")


# =============================================================================
# WELFARE DERIVATIONS
# =============================================================================

def derive_welfare_theorems() -> Derivation:
    """Explain the welfare theorems step by step."""
    steps = []
    
    steps.append(DerivationStep(
        1, "Setup",
        "Consider a pure exchange economy with n consumers, m goods.",
        "\\text{Endowments: } \\omega^i, \\text{ Preferences: } U^i(x^i)"
    ))
    
    steps.append(DerivationStep(
        2, "Pareto Efficiency Definition",
        "An allocation x* is Pareto efficient if no other feasible allocation makes someone better off without making anyone worse off.",
        "\\nexists x': U^i(x') \\geq U^i(x^*) \\forall i \\text{ with strict for some } i"
    ))
    
    steps.append(DerivationStep(
        3, "Competitive Equilibrium Definition",
        "A CE is prices p* and allocations x* such that: (1) each consumer maximizes utility given budget, (2) markets clear.",
        "x^{i*} = \\arg\\max U^i(x^i) \\text{ s.t. } p \\cdot x^i \\leq p \\cdot \\omega^i"
    ))
    
    steps.append(DerivationStep(
        4, "First Welfare Theorem",
        "If preferences are locally non-satiated, any CE allocation is Pareto efficient.",
        "\\text{CE} \\Rightarrow \\text{Pareto Efficient}"
    ))
    
    steps.append(DerivationStep(
        5, "FWT Proof Sketch",
        "Suppose CE allocation x* is not PE. Then exists x' Pareto-dominating x*. But if x'^i preferred to x*^i, then p·x'^i > p·ω^i (by utility max). Summing: p·Σx'^i > p·Σω^i. But feasibility requires Σx'^i ≤ Σω^i. Contradiction.",
        "\\text{Contradiction proves FWT}"
    ))
    
    steps.append(DerivationStep(
        6, "Second Welfare Theorem",
        "If preferences are convex and continuous, any PE allocation can be achieved as a CE with appropriate lump-sum transfers.",
        "\\forall x^{PE} \\exists T, p: (p, x^{PE}) \\text{ is CE with transfers } T"
    ))
    
    steps.append(DerivationStep(
        7, "SWT Implication",
        "Efficiency and equity can be separated: use markets for efficiency, transfers for distribution.",
        "\\text{Equity via transfers} + \\text{Efficiency via markets}"
    ))
    
    return Derivation("Welfare Theorems", steps)
