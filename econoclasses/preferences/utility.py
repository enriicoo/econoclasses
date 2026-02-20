"""
Core Utility class for preference analysis.
"""

import sympy as sp
import numpy as np
from typing import Optional, Dict, Tuple, Union, Literal
from dataclasses import dataclass

from ..core import X, Y, Px, Py, I, lam
from .forms import FORMS, get_form
from .solvers import (
    DemandSolution,
    MARSHALLIAN_SOLVERS, HICKSIAN_SOLVERS,
    marshallian_numerical, hicksian_numerical
)


class Utility:
    """
    A utility function with methods for demand analysis, elasticities, and decomposition.

    Parameters
    ----------
    form : str or sympy.Expr
        The utility function form. Use one of the built-in string names below,
        or pass a custom SymPy expression in terms of X and Y.

        Built-in forms and their keyword parameters
        -------------------------------------------
        'cobb-douglas'        U = X^α · Y^β
            alpha (float):  exponent on X  (default 0.5)
            beta  (float):  exponent on Y  (default 0.5)

        'perfect-substitutes' U = αX + βY
            alpha (float):  weight on X  (default 1)
            beta  (float):  weight on Y  (default 1)

        'perfect-complements' U = min(αX, βY)
            alpha (float):  coefficient on X  (default 1)
            beta  (float):  coefficient on Y  (default 1)

        'ces'                 U = [α X^ρ + (1-α) Y^ρ]^(1/ρ)
            alpha (float):  share parameter, 0 < α < 1  (default 0.5)
            rho   (float):  substitution parameter ρ ≠ 0  (default 0.5)
                            ρ → 0 → Cobb-Douglas, ρ = 1 → perfect substitutes,
                            ρ → -∞ → perfect complements

        'quasilinear'         U = X + α·ln(Y)  or  U = α·ln(X) + Y
            alpha     (float):  coefficient on the log term  (default 1)
            numeraire (str):    'X' → X is linear, Y is log  (default 'X')
                                'Y' → Y is linear, X is log

        'stone-geary'         U = (X - γx)^α · (Y - γy)^β
            alpha   (float):  exponent on (X - γx)  (default 0.5)
            beta    (float):  exponent on (Y - γy)  (default 0.5)
            gamma_x (float):  subsistence quantity of X  (default 0)
            gamma_y (float):  subsistence quantity of Y  (default 0)

        'translog'            ln U = α₀ + αx·ln X + αy·ln Y + ½βxx·(ln X)² + ½βyy·(ln Y)² + βxy·ln X·ln Y
            alpha_0 (float):  intercept  (default 0)
            alpha_x (float):  first-order X term  (default 0.5)
            alpha_y (float):  first-order Y term  (default 0.5)
            beta_xx (float):  second-order X term  (default 0)
            beta_yy (float):  second-order Y term  (default 0)
            beta_xy (float):  cross term  (default 0)

    income : float
        Consumer's budget / income (default 100).
    price_x : float
        Price of good X (default 1).
    price_y : float
        Price of good Y (default 1).
    **params
        Form-specific parameters as listed above.

    Examples
    --------
    >>> u = Utility('cobb-douglas', alpha=0.3, beta=0.7, income=100, price_x=2, price_y=3)
    >>> u.marshallian_demand
    (15.0, 23.33...)

    >>> u = Utility('ces', alpha=0.5, rho=0.5, income=100, price_x=2, price_y=3)
    >>> u = Utility('quasilinear', alpha=2, numeraire='Y', income=50, price_x=1, price_y=2)
    >>> u = Utility('stone-geary', alpha=0.5, beta=0.5, gamma_x=5, gamma_y=3, income=100)
    """

    def __init__(
        self,
        form: Union[str, sp.Expr],
        income: float = 100.0,
        price_x: float = 1.0,
        price_y: float = 1.0,
        **params
    ):
        self.income = income
        self.price_x = price_x
        self.price_y = price_y
        self._params = params
        
        if isinstance(form, str):
            self.form_name = form
            self.expr = get_form(form, **params)
            self._has_analytical_solver = form in MARSHALLIAN_SOLVERS
        else:
            self.form_name = 'custom'
            self.expr = form
            self._has_analytical_solver = False
        
        self._utility_func = sp.lambdify(
            (X, Y), self.expr,
            modules=['numpy', {'Min': np.minimum, 'Max': np.maximum}]
        )
    
    # =========================================================================
    # CORE PROPERTIES
    # =========================================================================
    
    @property
    def marginal_utility_x(self) -> sp.Expr:
        return sp.diff(self.expr, X)
    
    @property
    def marginal_utility_y(self) -> sp.Expr:
        return sp.diff(self.expr, Y)
    
    @property
    def mrs(self) -> sp.Expr:
        """Marginal Rate of Substitution: MUx / MUy"""
        return sp.simplify(self.marginal_utility_x / self.marginal_utility_y)
    
    def mrs_at(self, x: float, y: float) -> float:
        return float(self.mrs.subs({X: x, Y: y}))
    
    def utility_at(self, x: float, y: float) -> float:
        return float(self._utility_func(x, y))

    def get_equation_latex(self) -> str:
        """Return LaTeX representation of the utility function."""
        return f"U(X,Y) = {sp.latex(self.expr)}"

    # =========================================================================
    # DEMAND FUNCTIONS
    # =========================================================================
    
    @property
    def marshallian_demand(self) -> Tuple[float, float]:
        return self._solve_marshallian().as_tuple()
    
    @property
    def marshallian_demand_solution(self) -> DemandSolution:
        return self._solve_marshallian()
    
    def demand_at_prices(self, price_x: float, price_y: float, 
                         income: Optional[float] = None) -> Tuple[float, float]:
        income = income if income is not None else self.income
        return self._solve_marshallian(price_x, price_y, income).as_tuple()
    
    def hicksian_demand(self, u_target: Optional[float] = None,
                        price_x: Optional[float] = None,
                        price_y: Optional[float] = None) -> Tuple[float, float]:
        if u_target is None:
            u_target = self.utility_at(*self.marshallian_demand)
        px = price_x if price_x is not None else self.price_x
        py = price_y if price_y is not None else self.price_y
        return self._solve_hicksian(u_target, px, py).as_tuple()
    
    def _solve_marshallian(self, price_x: Optional[float] = None,
                           price_y: Optional[float] = None,
                           income: Optional[float] = None) -> DemandSolution:
        px = price_x if price_x is not None else self.price_x
        py = price_y if price_y is not None else self.price_y
        inc = income if income is not None else self.income
        
        if self._has_analytical_solver:
            solver = MARSHALLIAN_SOLVERS[self.form_name]
            if self.form_name == 'cobb-douglas':
                return solver(self._params.get('alpha', 0.5),
                             self._params.get('beta', 0.5), px, py, inc)
            elif self.form_name == 'perfect-substitutes':
                return solver(self._params.get('alpha', 1),
                             self._params.get('beta', 1), px, py, inc)
            elif self.form_name == 'perfect-complements':
                return solver(self._params.get('alpha', 1),
                             self._params.get('beta', 1), px, py, inc)
            elif self.form_name == 'ces':
                return solver(self._params.get('alpha', 0.5),
                             self._params.get('rho', 0.5), px, py, inc)
            elif self.form_name == 'quasilinear':
                return solver(self._params.get('alpha', 1),
                             self._params.get('numeraire', 'X'), px, py, inc)
            elif self.form_name == 'stone-geary':
                return solver(self._params.get('alpha', 0.5),
                             self._params.get('beta', 0.5),
                             self._params.get('gamma_x', 0),
                             self._params.get('gamma_y', 0), px, py, inc)
        
        return marshallian_numerical(self._utility_func, px, py, inc)
    
    def _solve_hicksian(self, u_target: float, px: float, py: float) -> DemandSolution:
        if self.form_name in HICKSIAN_SOLVERS:
            solver = HICKSIAN_SOLVERS[self.form_name]
            if self.form_name == 'cobb-douglas':
                return solver(self._params.get('alpha', 0.5),
                             self._params.get('beta', 0.5), px, py, u_target)
            elif self.form_name == 'perfect-substitutes':
                return solver(self._params.get('alpha', 1),
                             self._params.get('beta', 1), px, py, u_target)
            elif self.form_name == 'perfect-complements':
                return solver(self._params.get('alpha', 1),
                             self._params.get('beta', 1), px, py, u_target)
            elif self.form_name == 'ces':
                return solver(self._params.get('alpha', 0.5),
                             self._params.get('rho', 0.5), px, py, u_target)
        
        return hicksian_numerical(self._utility_func, px, py, u_target)
    
    # =========================================================================
    # INDIRECT UTILITY AND EXPENDITURE
    # =========================================================================
    
    @property
    def indirect_utility(self) -> float:
        x, y = self.marshallian_demand
        return self.utility_at(x, y)
    
    def expenditure(self, u_target: Optional[float] = None) -> float:
        if u_target is None:
            u_target = self.indirect_utility
        x, y = self.hicksian_demand(u_target)
        return self.price_x * x + self.price_y * y
    
    # =========================================================================
    # ELASTICITIES
    # =========================================================================
    
    @property
    def marshallian_demand_symbolic(self) -> Optional[Dict[str, sp.Expr]]:
        if self.form_name == 'cobb-douglas':
            a = self._params.get('alpha', 0.5)
            b = self._params.get('beta', 0.5)
            share_x = sp.Rational(a / (a + b)).limit_denominator(1000)
            share_y = sp.Rational(b / (a + b)).limit_denominator(1000)
            return {'X': share_x * I / Px, 'Y': share_y * I / Py}
        elif self.form_name == 'stone-geary':
            a = self._params.get('alpha', 0.5)
            b = self._params.get('beta', 0.5)
            gx = self._params.get('gamma_x', 0)
            gy = self._params.get('gamma_y', 0)
            share_x = sp.Rational(a / (a + b)).limit_denominator(1000)
            share_y = sp.Rational(b / (a + b)).limit_denominator(1000)
            supernumerary = I - Px * gx - Py * gy
            return {'X': gx + share_x * supernumerary / Px,
                    'Y': gy + share_y * supernumerary / Py}
        return None
    
    def price_elasticity(self, good: Literal['X', 'Y'] = 'X') -> Union[sp.Expr, float]:
        symbolic = self.marshallian_demand_symbolic
        if symbolic is not None:
            demand = symbolic[good]
            price = Px if good == 'X' else Py
            return sp.simplify(sp.diff(demand, price) * (price / demand))
        return self._numerical_price_elasticity(good)
    
    def income_elasticity(self, good: Literal['X', 'Y'] = 'X') -> Union[sp.Expr, float]:
        symbolic = self.marshallian_demand_symbolic
        if symbolic is not None:
            demand = symbolic[good]
            return sp.simplify(sp.diff(demand, I) * (I / demand))
        return self._numerical_income_elasticity(good)
    
    def cross_price_elasticity(self, demand_good: Literal['X', 'Y'] = 'X') -> Union[sp.Expr, float]:
        symbolic = self.marshallian_demand_symbolic
        if symbolic is not None:
            demand = symbolic[demand_good]
            other_price = Py if demand_good == 'X' else Px
            return sp.simplify(sp.diff(demand, other_price) * (other_price / demand))
        return self._numerical_cross_elasticity(demand_good)
    
    def _numerical_price_elasticity(self, good: str, delta: float = 0.01) -> float:
        px, py = self.price_x, self.price_y
        x0, y0 = self.marshallian_demand
        q0 = x0 if good == 'X' else y0
        p0 = px if good == 'X' else py
        
        if good == 'X':
            x1, y1 = self.demand_at_prices(px * (1 + delta), py)
            q1 = x1
        else:
            x1, y1 = self.demand_at_prices(px, py * (1 + delta))
            q1 = y1
        
        dq = q1 - q0
        dp = p0 * delta
        return (dq / dp) * (p0 / q0) if q0 != 0 else 0
    
    def _numerical_income_elasticity(self, good: str, delta: float = 0.01) -> float:
        x0, y0 = self.marshallian_demand
        q0 = x0 if good == 'X' else y0
        x1, y1 = self.demand_at_prices(self.price_x, self.price_y, self.income * (1 + delta))
        q1 = x1 if good == 'X' else y1
        dq = q1 - q0
        di = self.income * delta
        return (dq / di) * (self.income / q0) if q0 != 0 else 0
    
    def _numerical_cross_elasticity(self, demand_good: str, delta: float = 0.01) -> float:
        px, py = self.price_x, self.price_y
        x0, y0 = self.marshallian_demand
        
        if demand_good == 'X':
            q0 = x0
            x1, y1 = self.demand_at_prices(px, py * (1 + delta))
            q1 = x1
            p0 = py
        else:
            q0 = y0
            x1, y1 = self.demand_at_prices(px * (1 + delta), py)
            q1 = y1
            p0 = px
        
        dq = q1 - q0
        dp = p0 * delta
        return (dq / dp) * (p0 / q0) if q0 != 0 else 0
    
    def classify_good(self, good: Literal['X', 'Y'] = 'X') -> str:
        eta = self.income_elasticity(good)
        if isinstance(eta, sp.Expr):
            eta = float(eta.subs({I: self.income, Px: self.price_x, Py: self.price_y}))
        
        if eta > 1:
            return 'Luxury'
        elif eta > 0:
            return 'Normal (Necessity)'
        elif eta < 0:
            return 'Inferior'
        else:
            return 'Unit elastic'
    
    # =========================================================================
    # SLUTSKY DECOMPOSITION
    # =========================================================================
    
    def slutsky_decomposition(
        self,
        new_price_x: Optional[float] = None,
        new_price_y: Optional[float] = None,
        good: Literal['X', 'Y'] = 'X'
    ) -> Dict[str, float]:
        px0, py0 = self.price_x, self.price_y
        x0, y0 = self.marshallian_demand
        u0 = self.utility_at(x0, y0)
        
        px1 = new_price_x if new_price_x is not None else px0
        py1 = new_price_y if new_price_y is not None else py0
        
        x1, y1 = self.demand_at_prices(px1, py1, self.income)
        x_h, y_h = self.hicksian_demand(u0, px1, py1)
        
        if good == 'X':
            total = x1 - x0
            substitution = x_h - x0
        else:
            total = y1 - y0
            substitution = y_h - y0
        
        income = total - substitution
        
        return {
            'total_effect': total,
            'substitution_effect': substitution,
            'income_effect': income,
            'original_bundle': (x0, y0),
            'compensated_bundle': (x_h, y_h),
            'new_bundle': (x1, y1)
        }
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def to_latex(self) -> str:
        return sp.latex(self.expr)
    
    def __repr__(self) -> str:
        if self.form_name != 'custom':
            params_str = ', '.join(f'{k}={v}' for k, v in self._params.items())
            return f"Utility('{self.form_name}', {params_str})"
        return f"Utility({self.expr})"
    
    def summary(self) -> str:
        x, y = self.marshallian_demand
        u = self.utility_at(x, y)
        return f"""Utility: {self.form_name}
  U = {self.expr}
Budget: I={self.income}, Px={self.price_x}, Py={self.price_y}
Optimal: X*={x:.4f}, Y*={y:.4f}, U*={u:.4f}
Price elasticity (X): {self.price_elasticity('X')}
Income elasticity (X): {self.income_elasticity('X')}
Classification: {self.classify_good('X')}"""
