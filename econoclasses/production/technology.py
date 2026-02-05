"""
Core ProductionFunction class.

Analogous to preferences/utility.py but for production.
"""

import sympy as sp
import numpy as np
from typing import Optional, Dict, Tuple, Union, Literal
from dataclasses import dataclass

from ..core import K, L, w, r, Q
from .forms import FORMS, get_form
from .solvers import (
    CostSolution, ProfitSolution,
    COST_MIN_SOLVERS, PROFIT_MAX_SOLVERS,
    cost_min_numerical, profit_max_numerical
)


class ProductionFunction:
    """
    A production function with methods for cost minimization, profit maximization,
    and returns to scale analysis.
    
    Examples
    --------
    >>> pf = ProductionFunction('cobb-douglas', alpha=0.3, A=2)
    >>> pf.output_at(10, 20)  # Q = 2 * 10^0.3 * 20^0.7
    
    >>> pf.cost_minimize(Q_target=100, wage=5, rental=10)
    >>> pf.profit_maximize(price=15, wage=5, rental=10)
    """
    
    def __init__(self, form: Union[str, sp.Expr], **params):
        """
        Initialize a production function.
        
        Parameters
        ----------
        form : str or sympy.Expr
            Either a named form ('cobb-douglas', 'ces', etc.) or a SymPy expression.
        **params
            Parameters for the production form (e.g., alpha, A, rho).
        """
        self._params = params
        
        if isinstance(form, str):
            self.form_name = form
            self.expr = get_form(form, **params)
            self._has_analytical_solver = form in COST_MIN_SOLVERS
        else:
            self.form_name = 'custom'
            self.expr = form
            self._has_analytical_solver = False
        
        # Create numerical evaluation function
        self._production_func = sp.lambdify(
            (K, L), self.expr,
            modules=['numpy', {'Min': np.minimum, 'Max': np.maximum}]
        )
    
    # =========================================================================
    # CORE PROPERTIES
    # =========================================================================
    
    @property
    def marginal_product_K(self) -> sp.Expr:
        """∂Q/∂K - Marginal product of capital."""
        return sp.diff(self.expr, K)
    
    @property
    def marginal_product_L(self) -> sp.Expr:
        """∂Q/∂L - Marginal product of labor."""
        return sp.diff(self.expr, L)
    
    @property
    def mrts(self) -> sp.Expr:
        """
        Marginal Rate of Technical Substitution: MP_L / MP_K
        
        Rate at which capital can be substituted for labor while maintaining output.
        """
        return sp.simplify(self.marginal_product_L / self.marginal_product_K)
    
    def mrts_at(self, k: float, l: float) -> float:
        """Evaluate MRTS at a specific point."""
        return float(self.mrts.subs({K: k, L: l}))
    
    def output_at(self, k: float, l: float) -> float:
        """Evaluate output at given inputs."""
        return float(self._production_func(k, l))
    
    def mp_k_at(self, k: float, l: float) -> float:
        """Marginal product of capital at a point."""
        return float(self.marginal_product_K.subs({K: k, L: l}))
    
    def mp_l_at(self, k: float, l: float) -> float:
        """Marginal product of labor at a point."""
        return float(self.marginal_product_L.subs({K: k, L: l}))
    
    # =========================================================================
    # RETURNS TO SCALE
    # =========================================================================
    
    @property
    def returns_to_scale(self) -> str:
        """
        Determine returns to scale by checking f(tK, tL) vs t×f(K, L).
        """
        t = sp.Symbol('t', positive=True)
        
        scaled = self.expr.subs({K: t*K, L: t*L})
        original_scaled = t * self.expr
        
        # Simplify and compare
        ratio = sp.simplify(scaled / original_scaled)
        
        # For Cobb-Douglas, this gives t^(α+β-1)
        # If ratio simplifies to 1, CRS
        if ratio == 1:
            return 'Constant'
        
        # Check if ratio > 1 or < 1 for t > 1
        test_ratio = float(ratio.subs(t, 2))
        if test_ratio > 1.001:
            return 'Increasing'
        elif test_ratio < 0.999:
            return 'Decreasing'
        else:
            return 'Constant'
    
    def scale_elasticity(self) -> float:
        """
        Elasticity of scale: ε = MP_K × K/Q + MP_L × L/Q
        
        ε = 1: CRS, ε > 1: IRS, ε < 1: DRS
        """
        elasticity = (self.marginal_product_K * K + self.marginal_product_L * L) / self.expr
        return sp.simplify(elasticity)
    
    # =========================================================================
    # COST MINIMIZATION
    # =========================================================================
    
    def cost_minimize(self, Q_target: float, wage: float, rental: float) -> CostSolution:
        """
        Find input combination that minimizes cost for a given output.
        
        Parameters
        ----------
        Q_target : float
            Target output level
        wage : float
            Wage rate (price of labor)
        rental : float
            Rental rate (price of capital)
        
        Returns
        -------
        CostSolution with K*, L*, total cost
        """
        if self._has_analytical_solver:
            solver = COST_MIN_SOLVERS[self.form_name]
            
            if self.form_name == 'cobb-douglas':
                return solver(self._params.get('alpha', 0.3),
                             self._params.get('A', 1),
                             wage, rental, Q_target)
            elif self.form_name == 'cobb-douglas-general':
                return solver(self._params.get('alpha', 0.3),
                             self._params.get('beta', 0.7),
                             self._params.get('A', 1),
                             wage, rental, Q_target)
            elif self.form_name == 'ces':
                return solver(self._params.get('alpha', 0.5),
                             self._params.get('rho', 0.5),
                             self._params.get('A', 1),
                             wage, rental, Q_target)
            elif self.form_name == 'leontief':
                return solver(self._params.get('alpha', 1),
                             self._params.get('beta', 1),
                             self._params.get('A', 1),
                             wage, rental, Q_target)
            elif self.form_name == 'linear':
                return solver(self._params.get('alpha', 1),
                             self._params.get('beta', 1),
                             self._params.get('A', 1),
                             wage, rental, Q_target)
        
        return cost_min_numerical(self._production_func, wage, rental, Q_target)
    
    def cost_function(self, Q_target: float, wage: float, rental: float) -> float:
        """
        C(Q, w, r) - Total cost as a function of output.
        """
        solution = self.cost_minimize(Q_target, wage, rental)
        return solution.cost
    
    def average_cost(self, Q_target: float, wage: float, rental: float) -> float:
        """AC(Q) = C(Q) / Q"""
        return self.cost_function(Q_target, wage, rental) / Q_target
    
    def marginal_cost(self, Q_target: float, wage: float, rental: float, 
                      delta: float = 0.01) -> float:
        """MC(Q) = ∂C/∂Q (numerical approximation)"""
        c1 = self.cost_function(Q_target, wage, rental)
        c2 = self.cost_function(Q_target * (1 + delta), wage, rental)
        return (c2 - c1) / (Q_target * delta)
    
    # =========================================================================
    # PROFIT MAXIMIZATION
    # =========================================================================
    
    def profit_maximize(self, price: float, wage: float, rental: float) -> ProfitSolution:
        """
        Find inputs that maximize profit: π = P×Q - w×L - r×K
        
        Parameters
        ----------
        price : float
            Output price
        wage : float
            Wage rate
        rental : float
            Rental rate
        
        Returns
        -------
        ProfitSolution with K*, L*, Q*, profit
        """
        if self.form_name in PROFIT_MAX_SOLVERS:
            solver = PROFIT_MAX_SOLVERS[self.form_name]
            
            if self.form_name == 'cobb-douglas':
                return solver(self._params.get('alpha', 0.3),
                             self._params.get('A', 1),
                             wage, rental, price)
            elif self.form_name == 'cobb-douglas-general':
                return solver(self._params.get('alpha', 0.3),
                             self._params.get('beta', 0.7),
                             self._params.get('A', 1),
                             wage, rental, price)
        
        return profit_max_numerical(self._production_func, wage, rental, price)
    
    # =========================================================================
    # ISOQUANTS
    # =========================================================================
    
    def isoquant(self, Q_level: float, K_range: Tuple[float, float],
                 n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute isoquant: combinations of (K, L) that produce Q_level.
        
        Returns (K_array, L_array) for plotting.
        """
        K_vals = np.linspace(K_range[0], K_range[1], n_points)
        L_vals = []
        
        for k in K_vals:
            # Solve Q(k, L) = Q_level for L
            try:
                # Binary search for L
                L_low, L_high = 0.001, 1000
                for _ in range(50):  # Max iterations
                    L_mid = (L_low + L_high) / 2
                    Q_mid = self.output_at(k, L_mid)
                    if Q_mid < Q_level:
                        L_low = L_mid
                    else:
                        L_high = L_mid
                    if abs(Q_mid - Q_level) < 1e-6:
                        break
                L_vals.append(L_mid)
            except:
                L_vals.append(np.nan)
        
        return K_vals, np.array(L_vals)
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def to_latex(self) -> str:
        return sp.latex(self.expr)
    
    def __repr__(self) -> str:
        if self.form_name != 'custom':
            params_str = ', '.join(f'{k}={v}' for k, v in self._params.items())
            return f"ProductionFunction('{self.form_name}', {params_str})"
        return f"ProductionFunction({self.expr})"
    
    def summary(self, wage: float = 5, rental: float = 10, Q: float = 100) -> str:
        """Print summary of the production function."""
        cost_sol = self.cost_minimize(Q, wage, rental)
        
        return f"""Production Function: {self.form_name}
  Q = {self.expr}
  Returns to scale: {self.returns_to_scale}

At w={wage}, r={rental}, Q={Q}:
  Cost-minimizing inputs: K*={cost_sol.K:.4f}, L*={cost_sol.L:.4f}
  Total cost: {cost_sol.cost:.4f}
  Average cost: {self.average_cost(Q, wage, rental):.4f}
  Marginal cost: {self.marginal_cost(Q, wage, rental):.4f}"""
