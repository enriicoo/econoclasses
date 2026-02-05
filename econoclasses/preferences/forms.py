"""
Utility function forms.

Each function returns a SymPy expression for the utility function.
These are pure functions - no state, no side effects.
"""

import sympy as sp
from ..core import X, Y


def cobb_douglas(alpha: float, beta: float) -> sp.Expr:
    """U = X^α × Y^β"""
    return X**sp.Rational(alpha).limit_denominator(1000) * Y**sp.Rational(beta).limit_denominator(1000)


def perfect_substitutes(alpha: float, beta: float) -> sp.Expr:
    """U = αX + βY"""
    return sp.Rational(alpha).limit_denominator(1000) * X + sp.Rational(beta).limit_denominator(1000) * Y


def perfect_complements(alpha: float, beta: float) -> sp.Expr:
    """U = min(αX, βY)"""
    return sp.Min(sp.Rational(alpha).limit_denominator(1000) * X, 
                  sp.Rational(beta).limit_denominator(1000) * Y)


def ces(alpha: float, rho: float) -> sp.Expr:
    """U = [α X^ρ + (1-α) Y^ρ]^(1/ρ)"""
    a = sp.Rational(alpha).limit_denominator(1000)
    r = sp.Rational(rho).limit_denominator(1000)
    return (a * X**r + (1 - a) * Y**r) ** (1/r)


def quasilinear(alpha: float = 1, numeraire: str = 'X') -> sp.Expr:
    """U = X + α ln(Y) or U = α ln(X) + Y"""
    a = sp.Rational(alpha).limit_denominator(1000)
    if numeraire == 'X':
        return X + a * sp.log(Y)
    else:
        return a * sp.log(X) + Y


def stone_geary(alpha: float, beta: float, gamma_x: float, gamma_y: float) -> sp.Expr:
    """U = (X - γx)^α × (Y - γy)^β"""
    a = sp.Rational(alpha).limit_denominator(1000)
    b = sp.Rational(beta).limit_denominator(1000)
    gx = sp.Rational(gamma_x).limit_denominator(1000)
    gy = sp.Rational(gamma_y).limit_denominator(1000)
    return (X - gx)**a * (Y - gy)**b


def translog(alpha_0: float, alpha_x: float, alpha_y: float,
             beta_xx: float = 0, beta_yy: float = 0, beta_xy: float = 0) -> sp.Expr:
    """ln(U) = α₀ + αx ln(X) + αy ln(Y) + ½βxx [ln(X)]² + ½βyy [ln(Y)]² + βxy ln(X)ln(Y)"""
    ln_x = sp.log(X)
    ln_y = sp.log(Y)
    ln_u = (alpha_0 + alpha_x * ln_x + alpha_y * ln_y +
            sp.Rational(1, 2) * beta_xx * ln_x**2 +
            sp.Rational(1, 2) * beta_yy * ln_y**2 +
            beta_xy * ln_x * ln_y)
    return sp.exp(ln_u)


# Registry
FORMS = {
    'cobb-douglas': {
        'func': cobb_douglas,
        'params': ['alpha', 'beta'],
        'defaults': {'alpha': 0.5, 'beta': 0.5},
    },
    'perfect-substitutes': {
        'func': perfect_substitutes,
        'params': ['alpha', 'beta'],
        'defaults': {'alpha': 1, 'beta': 1},
    },
    'perfect-complements': {
        'func': perfect_complements,
        'params': ['alpha', 'beta'],
        'defaults': {'alpha': 1, 'beta': 1},
    },
    'ces': {
        'func': ces,
        'params': ['alpha', 'rho'],
        'defaults': {'alpha': 0.5, 'rho': 0.5},
    },
    'quasilinear': {
        'func': quasilinear,
        'params': ['alpha', 'numeraire'],
        'defaults': {'alpha': 1, 'numeraire': 'X'},
    },
    'stone-geary': {
        'func': stone_geary,
        'params': ['alpha', 'beta', 'gamma_x', 'gamma_y'],
        'defaults': {'alpha': 0.5, 'beta': 0.5, 'gamma_x': 0, 'gamma_y': 0},
    },
    'translog': {
        'func': translog,
        'params': ['alpha_0', 'alpha_x', 'alpha_y', 'beta_xx', 'beta_yy', 'beta_xy'],
        'defaults': {'alpha_0': 0, 'alpha_x': 0.5, 'alpha_y': 0.5, 
                     'beta_xx': 0, 'beta_yy': 0, 'beta_xy': 0},
    },
}


def get_form(name: str, **params) -> sp.Expr:
    """Get a utility expression by name with parameters."""
    if name not in FORMS:
        available = ', '.join(FORMS.keys())
        raise ValueError(f"Unknown utility form '{name}'. Available: {available}")
    
    spec = FORMS[name]
    full_params = {**spec['defaults'], **params}
    func_params = {k: v for k, v in full_params.items() if k in spec['params']}
    return spec['func'](**func_params)
