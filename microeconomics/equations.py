# equations.py

import sympy as sp

def cobb_douglas(x_input, y_input, x_multiplier, y_multiplier, x_exponential, y_exponential, technology=1):
    """
    Cobb-Douglas function Z = technology * x_input^x_exponential * y_input^y_exponential

    :param x_input: Quantity of input X (Sympy Symbol or numerical value)
    :param y_input: Quantity of input Y (Sympy Symbol or numerical value)
    :param x_multiplier: Multiplier for input X (float or Sympy Rational)
    :param y_multiplier: Multiplier for input Y (float or Sympy Rational)
    :param x_exponential: Exponent for input X (float or Sympy Rational)
    :param y_exponential: Exponent for input Y (float or Sympy Rational)
    :param technology: Technology parameter A (float or Sympy Rational)
    :return: Sympy expression for the Cobb-Douglas function
    """
    return technology * (
            (x_multiplier * x_input) ** x_exponential *
            (y_multiplier * y_input) ** y_exponential)

def substitutes(x_input, y_input, x_multiplier, y_multiplier, x_exponential, y_exponential, technology=1):
    """
    Perfect substitutes function Z = technology * (x_multiplier * x_input + y_multiplier * y_input)

    :param x_input: Quantity of input X (Sympy Symbol or numerical value)
    :param y_input: Quantity of input Y (Sympy Symbol or numerical value)
    :param x_multiplier: Multiplier for input X (float or Sympy Rational)
    :param y_multiplier: Multiplier for input Y (float or Sympy Rational)
    :param x_exponential: Exponent for input X (usually 1)
    :param y_exponential: Exponent for input Y (usually 1)
    :param technology: Technology parameter A (float or Sympy Rational)
    :return: Sympy expression for the Perfect Substitutes function
    """
    return technology * (
            (x_multiplier * x_input) ** x_exponential +
            (y_multiplier * y_input) ** y_exponential)


def complements(x_input, y_input, x_multiplier, y_multiplier, x_exponential, y_exponential, technology=1):
    """
    Perfect complements function Z = technology * min(x_multiplier * x_input, y_multiplier * y_input)

    :param x_input: Quantity of input X (Sympy Symbol or numerical value)
    :param y_input: Quantity of input Y (Sympy Symbol or numerical value)
    :param x_multiplier: Multiplier for input X (float or Sympy Rational)
    :param y_multiplier: Multiplier for input Y (float or Sympy Rational)
    :param x_exponential: Exponent for input X (float or Sympy Rational)
    :param y_exponential: Exponent for input Y (float or Sympy Rational)
    :param technology: Technology parameter A (float or Sympy Rational)
    :return: Sympy expression for the Perfect Complements function
    """
    return technology * sp.Min((x_multiplier * x_input) ** x_exponential,
                               (y_multiplier * y_input) ** y_exponential)


def ces(x_input, y_input, x_multiplier, y_multiplier, x_exponential, y_exponential, technology=1, rho=0, alpha=0.5):
    """
    CES (Constant Elasticity of Substitution) function:

    Z = technology * [ alpha * (x_multiplier * x_input) ** rho + (1 - alpha) * (y_multiplier * y_input) ** rho ] ** (1 / rho)

    For rho = 0, the function typically approaches the Cobb-Douglas form.

    :param x_input: Quantity of input X (Sympy Symbol or numerical value)
    :param y_input: Quantity of input Y (Sympy Symbol or numerical value)
    :param x_multiplier: Multiplier for input X (float or Sympy Rational)
    :param y_multiplier: Multiplier for input Y (float or Sympy Rational)
    :param x_exponential: Exponent for input X (float or Sympy Rational; not directly used in standard CES)
    :param y_exponential: Exponent for input Y (float or Sympy Rational; not directly used in standard CES)
    :param technology: Technology parameter A (float or Sympy Rational)
    :param rho: Substitution parameter (float or Sympy Rational); governs the elasticity of substitution
    :param alpha: Distribution parameter for input X (float between 0 and 1)
    :return: Sympy expression for the CES function
    """
    term_x = alpha * (x_multiplier * x_input) ** rho
    term_y = (1 - alpha) * (y_multiplier * y_input) ** rho
    if rho == 0:
        # Handle the Cobb-Douglas case as rho approaches 0
        return technology * (x_multiplier * x_input) ** (alpha * x_exponential) * (y_multiplier * y_input) ** (
                    (1 - alpha) * y_exponential)
    else:
        return technology * (term_x + term_y) ** (1 / rho)


def stone_geary(x_input, y_input, x_multiplier, y_multiplier, x_exponential, y_exponential, technology=1, gamma_x=0,
                gamma_y=0):
    """
    Stone-Geary function:
    Z = technology * (x_multiplier * (x_input - gamma_x)) ** x_exponential * (y_multiplier * (y_input - gamma_y)) ** y_exponential

    :param x_input: Quantity of input X (Sympy Symbol or numerical value)
    :param y_input: Quantity of input Y (Sympy Symbol or numerical value)
    :param x_multiplier: Multiplier for input X (float or Sympy Rational)
    :param y_multiplier: Multiplier for input Y (float or Sympy Rational)
    :param x_exponential: Exponent for input X (float or Sympy Rational)
    :param y_exponential: Exponent for input Y (float or Sympy Rational)
    :param technology: Technology parameter A (float or Sympy Rational)
    :param gamma_x: Subsistence (minimum required) quantity for X (float or Sympy Rational)
    :param gamma_y: Subsistence (minimum required) quantity for Y (float or Sympy Rational)
    :return: Sympy expression for the Stone-Geary function
    """
    return technology * (x_multiplier * (x_input - gamma_x)) ** x_exponential * (
                y_multiplier * (y_input - gamma_y)) ** y_exponential


def translog(x_input, y_input, x_multiplier, y_multiplier, technology=0, beta_xx=0, beta_yy=0, beta_xy=0):
    """
    Translog function:
    ln(Z) = technology + x_multiplier * ln(x_input) + y_multiplier * ln(y_input)
            + 0.5 * beta_xx * [ln(x_input)]^2 + 0.5 * beta_yy * [ln(y_input)]^2 + beta_xy * ln(x_input) * ln(y_input)
    Z = exp(ln(Z))

    Note: The parameters x_exponential and y_exponential are not used in the standard Translog formulation.

    :param x_input: Quantity of input X (Sympy Symbol or numerical value)
    :param y_input: Quantity of input Y (Sympy Symbol or numerical value)
    :param x_multiplier: Coefficient for ln(x_input) (float or Sympy Rational)
    :param y_multiplier: Coefficient for ln(y_input) (float or Sympy Rational)
    :param technology: Constant term (technology) (float or Sympy Rational)
    :param beta_xx: Coefficient for [ln(x_input)]^2 (float or Sympy Rational)
    :param beta_yy: Coefficient for [ln(y_input)]^2 (float or Sympy Rational)
    :param beta_xy: Coefficient for ln(x_input) * ln(y_input) (float or Sympy Rational)
    :return: Sympy expression for the Translog function
    """
    ln_x = sp.log(x_input)
    ln_y = sp.log(y_input)
    ln_z = technology + x_multiplier * ln_x + y_multiplier * ln_y \
           + 0.5 * beta_xx * ln_x ** 2 + 0.5 * beta_yy * ln_y ** 2 + beta_xy * ln_x * ln_y
    return sp.exp(ln_z)

def quasilinear(x_input, y_input, x_multiplier, y_multiplier, x_exponential, y_exponential, technology=0):
    """
    Quasilinear utility function:
    Z = technology + x_multiplier * (x_input ** x_exponential) + y_multiplier * (sp.log(y_input) ** y_exponential)

    Note: Typically one of the exponents is set to 1 (often x_exponential = 1), while y_exponential
    controls the curvature of the logarithmic term.

    :param x_input: Quantity of input X (Sympy Symbol or numerical value)
    :param y_input: Quantity of input Y (Sympy Symbol or numerical value)
    :param x_multiplier: Multiplier for the linear term in X (float or Sympy Rational)
    :param y_multiplier: Multiplier for the logarithmic term in Y (float or Sympy Rational)
    :param x_exponential: Exponent for x_input (float or Sympy Rational)
    :param y_exponential: Exponent for the logarithmic term of y_input (float or Sympy Rational)
    :param technology: Constant term (technology) (float or Sympy Rational)
    :return: Sympy expression for the Quasilinear function
    """
    term_x = x_multiplier * x_input ** x_exponential
    term_y = y_multiplier * sp.log(y_input) ** y_exponential
    return technology + term_x + term_y
