"""
Production function forms.

Each function returns a SymPy expression for output as a function of inputs.
Mirrors preferences/forms.py structure.
"""

import sympy as sp
from ..core import K, L


def cobb_douglas(alpha: float, A: float = 1) -> sp.Expr:
    """
    Q = A × K^α × L^(1-α)
    
    Constant returns to scale when exponents sum to 1.
    
    Parameters
    ----------
    alpha : float
        Capital share (0 < α < 1)
    A : float
        Total factor productivity
    """
    a = sp.Rational(alpha).limit_denominator(1000)
    return A * K**a * L**(1 - a)


def cobb_douglas_general(alpha: float, beta: float, A: float = 1) -> sp.Expr:
    """
    Q = A × K^α × L^β
    
    Returns to scale:
    - α + β < 1: Decreasing
    - α + β = 1: Constant  
    - α + β > 1: Increasing
    """
    a = sp.Rational(alpha).limit_denominator(1000)
    b = sp.Rational(beta).limit_denominator(1000)
    return A * K**a * L**b


def ces(alpha: float, rho: float, A: float = 1) -> sp.Expr:
    """
    Q = A × [α K^ρ + (1-α) L^ρ]^(1/ρ)
    
    Elasticity of substitution: σ = 1/(1-ρ)
    
    Special cases:
    - ρ → 0: Cobb-Douglas
    - ρ = 1: Perfect substitutes (linear)
    - ρ → -∞: Leontief (perfect complements)
    """
    a = sp.Rational(alpha).limit_denominator(1000)
    r = sp.Rational(rho).limit_denominator(1000)
    return A * (a * K**r + (1 - a) * L**r) ** (1/r)


def leontief(alpha: float, beta: float, A: float = 1) -> sp.Expr:
    """
    Q = A × min(αK, βL)
    
    Fixed proportions technology. Inputs must be used in ratio β/α.
    """
    a = sp.Rational(alpha).limit_denominator(1000)
    b = sp.Rational(beta).limit_denominator(1000)
    return A * sp.Min(a * K, b * L)


def linear(alpha: float, beta: float, A: float = 1) -> sp.Expr:
    """
    Q = A × (αK + βL)
    
    Perfect substitutes in production.
    """
    a = sp.Rational(alpha).limit_denominator(1000)
    b = sp.Rational(beta).limit_denominator(1000)
    return A * (a * K + b * L)


def quadratic(a: float, b: float, c: float, d: float = 0, e: float = 0, f: float = 0) -> sp.Expr:
    """
    Q = a + b×K + c×L + d×K² + e×L² + f×K×L
    
    Flexible form for short-run analysis.
    """
    return a + b*K + c*L + d*K**2 + e*L**2 + f*K*L


# Registry mapping names to functions
FORMS = {
    'cobb-douglas': {
        'func': cobb_douglas,
        'params': ['alpha', 'A'],
        'defaults': {'alpha': 0.3, 'A': 1},
    },
    'cobb-douglas-general': {
        'func': cobb_douglas_general,
        'params': ['alpha', 'beta', 'A'],
        'defaults': {'alpha': 0.3, 'beta': 0.7, 'A': 1},
    },
    'ces': {
        'func': ces,
        'params': ['alpha', 'rho', 'A'],
        'defaults': {'alpha': 0.5, 'rho': 0.5, 'A': 1},
    },
    'leontief': {
        'func': leontief,
        'params': ['alpha', 'beta', 'A'],
        'defaults': {'alpha': 1, 'beta': 1, 'A': 1},
    },
    'linear': {
        'func': linear,
        'params': ['alpha', 'beta', 'A'],
        'defaults': {'alpha': 1, 'beta': 1, 'A': 1},
    },
}


def get_form(name: str, **params) -> sp.Expr:
    """Get a production function expression by name."""
    if name not in FORMS:
        available = ', '.join(FORMS.keys())
        raise ValueError(f"Unknown production form '{name}'. Available: {available}")
    
    spec = FORMS[name]
    full_params = {**spec['defaults'], **params}
    func_params = {k: v for k, v in full_params.items() if k in spec['params']}
    return spec['func'](**func_params)
