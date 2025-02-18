import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import Optional, Literal, Tuple
from abc import ABC, abstractmethod

from microeconomics.equations import cobb_douglas, substitutes, complements

class BaseUtilityCurve(ABC):
    """
    Abstract base class for utility functions.
    Defines methods for calculating utility, demand functions, and elasticities.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        scale: float = 1.0,
        good_x: float = 5.0,
        good_y: float = 5.0,
        income: float = 100.0,
        price_x: float = 1.0,
        price_y: float = 1.0
    ):
        self.alpha = alpha if alpha is not None else None
        self.beta = beta if beta is not None else None
        self.scale = scale
        self.good_x = good_x
        self.good_y = good_y
        self.income = income
        self.price_x = price_x
        self.price_y = price_y

        # Preventing errors
        if (isinstance(self, PerfectComplementsUtilityCurve) and self.alpha == 0 or
                isinstance(self, PerfectComplementsUtilityCurve) and self.beta == 0):
            raise ValueError("Both alpha and beta must be strictly positive for perfect complements.")

        # Define symbolic variables
        self.X, self.Y = sp.symbols('X Y', real=True, positive=True)
        self.U = sp.Symbol('U', real=True, positive=True)
        self.I = sp.Symbol('I', real=True, positive=True)  # Income
        self.Px = sp.Symbol('P_x', real=True, positive=True)  # Price of X
        self.Py = sp.Symbol('P_y', real=True, positive=True)  # Price of Y

        # Initialize the symbolic utility function
        self.utility_expr = self.define_utility_function()

        # Substitute scale parameter for numerical evaluation
        self.numeric_expr = self.utility_expr.subs({self.U: self.scale})

        # Convert symbolic expression to a numerical function for plotting
        self.utility_func = sp.lambdify((self.X, self.Y),self.numeric_expr,
                                     modules=['numpy', {'Min': np.minimum, 'Max': np.maximum}])

        # Derive demand functions
        self.marshallian_demand = self.calculate_marshallian_demand()
        self.utility = self.calculate_utility(self.marshallian_demand[0], self.marshallian_demand[1])

        # Calculate elasticities
        self.price_elasticity_x = self.calculate_price_elasticity('X')
        self.price_elasticity_y = self.calculate_price_elasticity('Y')
        self.income_elasticity_x = self.calculate_income_elasticity('X')
        self.income_elasticity_y = self.calculate_income_elasticity('Y')
        self.cross_price_elasticity = self.calculate_cross_price_elasticity()

        # Classify goods
        self.classification_x = self.classify_good(self.income_elasticity_x)
        self.classification_y = self.classify_good(self.income_elasticity_y)

    @abstractmethod
    def define_utility_function(self):
        pass

    @abstractmethod
    def calculate_marshallian_demand(self, price_x: Optional[float] = None, price_y: Optional[float] = None,
                                     income: Optional[float] = None) -> Tuple[float, float]:
        pass

    @abstractmethod
    def calculate_hicksian_demand(self, u_target: float, price_x: float, price_y: float) -> Tuple[float, float]:
        """
        Return (xH, yH) that achieves utility U_target
        at prices (price_x, price_y) with minimum cost (i.e. Hicksian demand).
        """
        pass

    def slutsky_substitution(self, new_price_x: float, new_price_y: Optional[float] = None,
            new_income: Optional[float] = None, related_to: Literal['X', 'Y'] = 'X') -> Tuple[float, float, float]:
        """
        Returns a 3-tuple of the Slutsky decomposition for the chosen good (X or Y):
          (total_effect, income_effect, substitution_effect).

        :param new_price_x: The new price of good X.
        :param new_price_y: (Optional) The new price of good Y; if None, uses self.price_y.
        :param new_income:  (Optional) The new income; if None, uses self.income.
        :param related_to: 'X' or 'Y'; which good we are decomposing the price change effect for.
        """
        old_px = self.price_x
        old_py = self.price_y
        old_i = self.income
        px_1 = new_price_x
        py_1 = new_price_y if new_price_y is not None else self.price_y
        i_1 = new_income if new_income is not None else self.income

        # Original consumption bundle and its utility level
        x0, y0 = self.calculate_marshallian_demand(price_x=old_px, price_y=old_py, income=old_i)
        u_0 = float(self.utility_expr.subs({self.X: x0, self.Y: y0}))

        # New consumption bundle after the price (and income) change
        x1, y1 = self.calculate_marshallian_demand(price_x=px_1, price_y=py_1, income=i_1)

        # Compensated (Hicksian) demand at new prices keeping the utility at u_0
        x_h, y_h = self.calculate_hicksian_demand(u_0, px_1, py_1)

        if related_to == 'X':
            total_effect = x1 - x0
            substitution_effect = x_h - x0
            income_effect = total_effect - substitution_effect
        elif related_to == 'Y':
            total_effect = y1 - y0
            substitution_effect = y_h - y0
            income_effect = total_effect - substitution_effect
        else:
            raise ValueError("related_to must be 'X' or 'Y'")

        return total_effect, income_effect, substitution_effect

    def calculate_price_elasticity(self, good: str):
        symbolic_x_demand = self.alpha * self.I / self.Px
        symbolic_y_demand = self.beta * self.I / self.Py
        if good == 'X':
            demand = symbolic_x_demand
            price = self.Px
        elif good == 'Y':
            demand = symbolic_y_demand
            price = self.Py
        else:
            raise ValueError("Good must be 'X' or 'Y'.")
        elasticity = sp.diff(demand, price) * (price / demand)
        elasticity = sp.simplify(elasticity)
        return elasticity

    def calculate_income_elasticity(self, good: str):
        symbolic_x_demand = self.alpha * self.I / self.Px
        symbolic_y_demand = self.beta * self.I / self.Py
        if good == 'X':
            demand = symbolic_x_demand
        elif good == 'Y':
            demand = symbolic_y_demand
        else:
            raise ValueError("Good must be 'X' or 'Y'.")
        elasticity = sp.diff(demand, self.I) * (self.I / demand)
        elasticity = sp.simplify(elasticity)
        return elasticity

    def calculate_cross_price_elasticity(self):
        symbolic_x_demand = self.alpha * self.I / self.Px
        symbolic_y_demand = self.beta * self.I / self.Py
        elasticity_x = sp.diff(symbolic_x_demand, self.Py) * (self.Py / symbolic_x_demand)
        elasticity_y = sp.diff(symbolic_y_demand, self.Px) * (self.Px / symbolic_y_demand)
        elasticity_x = sp.simplify(elasticity_x)
        elasticity_y = sp.simplify(elasticity_y)
        return elasticity_x, elasticity_y

    def classify_good(self, income_elasticity):
        elasticity_value = income_elasticity.subs({self.I: self.income, self.Px: self.price_x, self.Py: self.price_y})
        elasticity_value = float(elasticity_value)

        if elasticity_value > 1:
            return 'Luxury Good'
        elif elasticity_value > 0:
            return 'Normal Good'
        elif elasticity_value < 0:
            return 'Inferior Good'
        else:
            return 'Necessity'

    def calculate_utility(self, x: float, y: float) -> float:
        # Evaluate the original symbolic expression, not just numeric_expr
        # so we can do deeper simplification.
        u_expr = self._coerce(self.utility_expr.subs({self.X: x, self.Y: y}))
        return u_expr

    def get_utility_function_latex(self, exponent_format='fraction') -> str:
        return sp.latex(self.utility_expr, root_notation=(exponent_format == 'root'))

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        cmap: str = 'viridis',
        title: Optional[str] = None,
        label: bool = False,
        label_position: Optional[Literal['first', 'mid', 'last']] = 'mid',
        label_direction: Optional[Literal['dynamic', 'horizontal', 'vertical']] = 'dynamic',
        show_colorbar: bool = True,
        show_equation: bool = False,
        num_curves: int = 5,
        x_limit: Optional[float] = None,
        y_limit: Optional[float] = None,
        show_prices: bool = False,
        show_optima: bool = False,
        box_position: Tuple[str, str] = ('low','right'),
        show_restriction: bool = False,
        **kwargs):
        """
        Plots the indifference curves for the utility function.

        :param ax: Matplotlib Axes object to plot on.
        :param cmap: Colormap to use.
        :param title: Title for the plot.
        :param label: If True, labels are added to the indifference curves.
        :param label_position: Position of the labels ('first', 'mid', 'last').
        :param label_direction: Direction of the labels ('dynamic', 'horizontal', 'vertical').
        :param show_colorbar: If True, displays a colorbar.
        :param show_equation: If True, shows the utility function equation in the title.
        :param num_curves: Number of indifference curves to plot.
        :param x_limit: Maximum value for X-axis.
        :param y_limit: Maximum value for Y-axis.
        :param show_prices: If True, a small legend-like box with "Px=?? | Py=??" is added on the plot.
        :param show_optima: If True, a small legend-like box with "X*=?? | Y*=??" is added on the plot.
        :param box_position: Tuple indicating the vertical ("low"/"high") and horizontal ("left"/"right") position of the box.
        :param show_restriction: If True, plots the budget constraint line.
        """
        x_star, y_star = self.marshallian_demand

        # Handle automatic limits if not provided
        if x_limit is None or y_limit is None:
            if isinstance(self, PerfectSubstitutesUtilityCurve):
                a_px, b_py = self.alpha / self.price_x, self.beta / self.price_y
                x_opt, y_opt = ((self.income / (2 * self.price_x), self.income / (2 * self.price_y))
                                if abs(a_px - b_py) < 1e-9 else (self.income / self.price_x, 0)
                                if a_px > b_py else (0, self.income / self.price_y))
                u_opt = self.alpha * x_opt + self.beta * y_opt
                x_limit = max(x_limit or 0, u_opt / self.alpha if self.alpha else 0)
                y_limit = max(y_limit or 0, u_opt / self.beta if self.beta else 0)
                if self.alpha == 0 or self.beta == 0:
                    x_limit = max(x_star, y_star) * 2 if x_star == 0 else x_star * 2
                    y_limit = x_limit if y_star == 0 else y_star * 2
            elif isinstance(self, PerfectComplementsUtilityCurve):
                u_opt = min(self.alpha * x_star, self.beta * y_star)
                if u_opt > 0:
                    x_limit = 2 * x_star
                    y_limit = 2 * y_star
                else:
                    x_limit = x_limit or self.income / self.price_x
                    y_limit = y_limit or self.income / self.price_y
                if self.alpha == 0 or self.beta == 0:
                    x_limit = max(x_star, y_star) * 2 if x_star == 0 else x_star * 2
                    y_limit = x_limit if y_star == 0 else y_star * 2
            else:
                x_limit = x_limit or (max(x_star, y_star) * 2 if x_star == 0 else x_star * 2)
                y_limit = y_limit or (x_limit if y_star == 0 else y_star * 2)
        x = np.linspace(0.1, x_limit, 300)
        y = np.linspace(0.1, y_limit, 300)
        x_grid, y_grid = np.meshgrid(x, y)
        u = self.utility_func(x_grid, y_grid)
        ax = ax if ax is not None else plt.subplots(figsize=(8, 6))[1]
        levels = np.linspace(self.utility * 0.5, self.utility * 1.5, num_curves)

        # Display the utility gradient as background
        if show_colorbar:
            cax = ax.imshow(u, extent=(0.1, x_limit, 0.1, y_limit), origin='lower', cmap=cmap, aspect='auto')
            cbar = plt.colorbar(cax, ax=ax)
            cbar.ax.set_title('Utility', fontsize=10, )

        # Generate indifference curve colors from the colormap with the generated colors
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(np.linspace(0.6, 1.0, len(levels)))
        contours = ax.contour(x_grid, y_grid, u, levels=levels, colors=colors)
        # Plot labels
        if label:
            middle_isoquant_idx = len(levels) // 2
            label_points = []
            for i, collection in enumerate(contours.collections):
                if i == middle_isoquant_idx: # Only label the "middle" isoquant
                    for path in collection.get_paths():
                        vertices = path.vertices
                        if label_position == 'first':
                            idx = len(vertices) // 4
                        elif label_position == 'last':
                            idx = (len(vertices) // 4)*3
                        else:  # 'mid' or anything else
                            idx = len(vertices) // 2
                        label_points.append(vertices[idx])
            labels = ax.clabel(contours, inline=True, fmt="%1.2f", fontsize=10, manual=label_points,
                               colors=[colors[-1]])
            if label_direction == 'horizontal':
                for label in labels:
                    label.set_rotation(0)
            elif label_direction == 'vertical':
                for label in labels:
                    label.set_rotation(90)

        # Restriction line
        if show_restriction:
            x_int = self.income / self.price_x
            x_array = np.linspace(0.1, min(x_int, x_limit), 300)
            y_array = (self.income - self.price_x * x_array) / self.price_y
            y_array = np.where((y_array >= 0.1) & (y_array <= y_limit), y_array, np.nan)
            mid_color = cmap_obj(0.5)
            inv_color = (1 - mid_color[0], 1 - mid_color[1], 1 - mid_color[2])
            ax.plot(x_array, y_array, color=inv_color, linewidth=2, ls='--', label='Budget Constraint')
            x_d, y_d = self.marshallian_demand
            x_d, y_d = [i + 0.2 if i == 0 else i for i in [x_d, y_d]]
            x_d, y_d = [d - 0.2 if d == limit else d for d, limit in [(x_d, x_limit), (y_d, y_limit)]]
            ax.plot(x_d, y_d, marker='o', color=inv_color, markersize=8, label='Optimal Bundle')

        # Box for prices and optimal bundle
        if show_prices or show_optima:
            text_lines = []
            if show_prices:
                left_price = f"Px = {self.price_x:.2f}"
                right_price = f"Py = {self.price_y:.2f}"
                price_text = f"{left_price:<10} | {right_price}"
                text_lines.append(price_text)
            if show_optima:
                x_star, y_star = self.marshallian_demand
                left_opt = f"X* = {x_star:.2f}"
                right_opt = f"Y* = {y_star:.2f}"
                optima_text = f"{left_opt:<10} | {right_opt}"
                text_lines.append(optima_text)
            info_box_text = "\n".join(text_lines)
            positions_map = {('low', 'left'): {'x': 0.02, 'y': 0.02, 'ha': 'left', 'va': 'bottom'},
                             ('low', 'right'): {'x': 0.98, 'y': 0.02, 'ha': 'right', 'va': 'bottom'},
                             ('high', 'left'): {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va': 'top'},
                             ('high', 'right'): {'x': 0.98, 'y': 0.98, 'ha': 'right', 'va': 'top'}}
            pos_args = positions_map.get(box_position, positions_map[('low', 'right')]) # fallback to (low, right)
            ax.text(pos_args['x'], pos_args['y'], info_box_text, transform=ax.transAxes, fontsize=10, ha=pos_args['ha'],
                    va=pos_args['va'], bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        ax.set_xlabel('Quantity of Good X')
        ax.set_ylabel('Quantity of Good Y')
        if show_equation and not title:
            eq_str = self.get_utility_function_latex()
            ax.set_title(f"${eq_str}$")
        elif title:
            ax.set_title(title)
        else:
            ax.set_title('Indifference Curves')
        ax.grid(True)

    def _coerce(self, expr, decimals=8):
        """
        Minimal helper to simplify expr and convert to int if possible,
        or to a float if not. Avoids 5.000000000000001, etc.
        """
        return float(round(expr.evalf(), decimals))

class CobbDouglasUtilityCurve(BaseUtilityCurve):
    """
    Class for Cobb-Douglas utility functions.
    """

    def define_utility_function(self):
        # Import utility function from equations.py
        return cobb_douglas(x_input=self.X, y_input=self.Y, x_multiplier=1, y_multiplier=1,
                            x_exponential=self.alpha, y_exponential=self.beta, technology=self.scale)

    def calculate_marshallian_demand(self, price_x: Optional[float] = None, price_y: Optional[float] = None,
                                     income: Optional[float] = None) -> Tuple[float, float]:
        price_x = price_x if price_x is not None else self.price_x
        price_y = price_y if price_y is not None else self.price_y
        income  = income  if income  is not None else self.income

        x_expr = (self.alpha / (self.alpha + self.beta)) * (income / price_x)
        y_expr = (self.beta / (self.alpha + self.beta)) * (income / price_y)
        return x_expr, y_expr

    def calculate_hicksian_demand(self, u_target: float, price_x: float, price_y: float) -> Tuple[float, float]:
        """
        From standard Lagrangian or ratio arguments, we get:
            Y / X = (beta / alpha) * (P_x / P_y)
            X^(alpha+beta) * (Y/X)^beta = U_bar
        so the solution is:
            X = [ U_bar / ( (beta/alpha * Px/Py)^beta ) ] ^ (1/(alpha+beta))
            Y = (beta/alpha) * (Px/Py) * X
        """
        # Symbolic utility at marshallian demands:
        x_m = self.alpha * self.I / self.Px
        y_m = self.beta * self.I / self.Py
        u_expr = self.utility_expr.subs({self.X: x_m, self.Y: y_m})

        # Symbolic ratio & Hicksian demands:
        ratio = (self.beta / self.alpha) * (self.Px / self.Py)
        x_star = (u_expr / (ratio ** self.beta)) ** (1 / (self.alpha + self.beta))
        y_star = ratio * x_star

        x_val = self._coerce(x_star.subs({self.I: self.income, self.Px: self.price_x, self.Py: self.price_y}))
        y_val = self._coerce(y_star.subs({self.I: self.income, self.Px: self.price_x, self.Py: self.price_y}))
        return x_val, y_val

    def get_utility_function_latex(self, exponent_format='fraction'):
        x = sp.latex(self.X)
        y = sp.latex(self.Y)

        alpha_expr = sp.Rational(self.alpha).limit_denominator(1000)
        beta_expr = sp.Rational(self.beta).limit_denominator(1000)

        if exponent_format == 'root':
            alpha_latex = sp.latex(alpha_expr)
            beta_latex = sp.latex(beta_expr)
        elif exponent_format == 'fraction':
            alpha_latex = sp.latex(alpha_expr, root_notation=False)
            beta_latex = sp.latex(beta_expr, root_notation=False)
        elif exponent_format == 'decimal':
            alpha_latex = f"{self.alpha:.3f}"
            beta_latex = f"{self.beta:.3f}"
        else:
            raise ValueError("Invalid exponent_format. Choose 'root', 'fraction', or 'decimal'.")

        term_x = f"{x}^{{{alpha_latex}}}"
        term_y = f"{y}^{{{beta_latex}}}"

        expr_latex = f"U = {term_x} \\times {term_y}"
        return expr_latex

class PerfectSubstitutesUtilityCurve(BaseUtilityCurve):
    """
    Class for perfect substitutes utility functions.
    """

    def define_utility_function(self):
        # Import utility function from equations.py
        return substitutes(x_input=self.X, y_input=self.Y, x_multiplier=self.alpha, y_multiplier=self.beta,
            x_exponential=1, y_exponential=1, technology=self.scale)

    def calculate_marshallian_demand(self, price_x: Optional[float] = None, price_y: Optional[float] = None,
                                     income: Optional[float] = None) -> Tuple[float, float]:
        price_x = price_x if price_x is not None else self.price_x
        price_y = price_y if price_y is not None else self.price_y
        income  = income  if income  is not None else self.income
        # Since we can't evaluate inequalities symbolically, we'll use numerical values
        alpha_px = self.alpha / price_x
        beta_py = self.beta / price_y

        if alpha_px > beta_py:
            x_demand = income / price_x
            y_demand = 0
        elif alpha_px < beta_py:
            x_demand = 0
            y_demand = income / price_y
        else:
            # Indifferent between goods
            x_demand = income / (price_x + price_y)
            y_demand = income / (price_x + price_y)
        return x_demand, y_demand

    def calculate_hicksian_demand(self, u_target: float, price_x: float, price_y: float) -> Tuple[float, float]:
        # For perfect substitutes, the Hicksian demand is similar to the Marshallian demand
        return self.calculate_marshallian_demand()

    def get_utility_function_latex(self, exponent_format='fraction'):
        x = sp.latex(self.X)
        y = sp.latex(self.Y)
        alpha_latex = sp.latex(self.alpha)
        beta_latex = sp.latex(self.beta)

        expr_latex = f"U = {alpha_latex} {x} + {beta_latex} {y}"
        return expr_latex

class PerfectComplementsUtilityCurve(BaseUtilityCurve):
    """
    Class for perfect complements utility functions.
    """
    def define_utility_function(self):
        # Import utility function from equations.py
        return complements(x_input=self.X, y_input=self.Y, x_multiplier=self.alpha, y_multiplier=self.beta,
                           x_exponential=1, y_exponential=1, technology=self.scale)
    def calculate_marshallian_demand(self, price_x: Optional[float] = None, price_y: Optional[float] = None,
                                     income: Optional[float] = None) -> Tuple[float, float]:
        price_x = price_x if price_x is not None else self.price_x
        price_y = price_y if price_y is not None else self.price_y
        income  = income  if income  is not None else self.income
        # The consumer will consume goods in fixed proportions
        proportion = self.alpha / self.beta
        x_demand = income / (price_x + proportion * price_y)
        y_demand = proportion * x_demand
        return x_demand, y_demand

    def calculate_hicksian_demand(self, u_target: float, price_x: float, price_y: float) -> Tuple[float, float]:
        u_bar = self.utility
        x_demand = u_bar / self.alpha
        y_demand = u_bar / self.beta
        return x_demand, y_demand

    def get_utility_function_latex(self, exponent_format='fraction'):
        x = sp.latex(self.X)
        y = sp.latex(self.Y)
        alpha_latex = sp.latex(self.alpha)
        beta_latex = sp.latex(self.beta)

        expr_latex = f"U = \\min\\{{ {alpha_latex} {x}, {beta_latex} {y} \\}}"
        return expr_latex

class UtilityCurve:
    """
    Factory class for creating utility curves of different types.
    Provides a simplified interface while leveraging existing specialized implementations.

    :param curve_type: A string literal indicating the utility-curve class to instantiate. Must be one of:
        ['cobb-douglas', 'perfect-substitutes', 'perfect-complements'].
    :param alpha: Parameter alpha for the utility function, if applicable.
    :param beta:  Parameter beta for the utility function, if applicable.
    :param scale: Scale or "technology" parameter often used in the equations.
    :param good_x: Initial guess or reference value for the quantity of good X.
    :param good_y: Initial guess or reference value for the quantity of good Y.
    :param income: Income of the consumer.
    :param price_x: Price of good X.
    :param price_y: Price of good Y.

    This class uses composition to hold an instance of one of the specialized classes:
      - CobbDouglasUtilityCurve
      - PerfectSubstitutesUtilityCurve
      - PerfectComplementsUtilityCurve

    All public methods and attributes of those specialized classes are accessible through this factory,
    thanks to the __getattr__ delegation mechanism. It effectively wraps and redirects all calls to
    the underlying specialized utility-curve object.
    """

    def __init__(
        self,
        curve_type: Literal['cobb-douglas', 'perfect-complements', 'perfect-substitutes'],
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        scale: float = 1.0,
        good_x: float = 5.0,
        good_y: float = 5.0,
        income: float = 100.0,
        price_x: float = 1.0,
        price_y: float = 1.0
    ):

        if curve_type == 'cobb-douglas':
            self._curve_instance = CobbDouglasUtilityCurve(alpha=alpha, beta=beta, scale=scale, good_x=good_x,
                good_y=good_y, income=income, price_x=price_x, price_y=price_y)
        elif curve_type == 'perfect-substitutes':
            self._curve_instance = PerfectSubstitutesUtilityCurve(alpha=alpha, beta=beta, scale=scale, good_x=good_x,
                good_y=good_y, income=income, price_x=price_x, price_y=price_y)
        elif curve_type == 'perfect-complements':
            self._curve_instance = PerfectComplementsUtilityCurve(alpha=alpha, beta=beta, scale=scale, good_x=good_x,
                good_y=good_y, income=income, price_x=price_x, price_y=price_y)
        else:
            raise ValueError(f"Invalid curve type '{curve_type}'. "
                             f"Must be one of ['cobb-douglas', 'perfect-substitutes', 'perfect-complements'].")

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying specialized utility-curve instance.
        This makes the factory act as a transparent wrapper. Any method or attribute
        that doesn't exist on the factory itself will be forwarded to the wrapped instance.
        """
        return getattr(self._curve_instance, name)
