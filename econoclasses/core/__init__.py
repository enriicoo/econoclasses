"""
Shared symbolic variables for econoclasses.

All modules import from here to ensure consistency.
"""

import sympy as sp

# =============================================================================
# GOODS AND QUANTITIES
# =============================================================================

X = sp.Symbol('X', real=True, positive=True)
Y = sp.Symbol('Y', real=True, positive=True)

# For production
K = sp.Symbol('K', real=True, positive=True)  # Capital
L = sp.Symbol('L', real=True, positive=True)  # Labor
Q = sp.Symbol('Q', real=True, positive=True)  # Output quantity

# =============================================================================
# PRICES
# =============================================================================

Px = sp.Symbol('P_x', real=True, positive=True)
Py = sp.Symbol('P_y', real=True, positive=True)
P = sp.Symbol('P', real=True, positive=True)   # Generic price

# Factor prices
w = sp.Symbol('w', real=True, positive=True)   # Wage
r = sp.Symbol('r', real=True, positive=True)   # Rental rate of capital

# =============================================================================
# INCOME AND BUDGET
# =============================================================================

I = sp.Symbol('I', real=True, positive=True)   # Income
M = sp.Symbol('M', real=True, positive=True)   # Alternative income symbol

# Endowments (for exchange economy)
omega_x = sp.Symbol('omega_x', real=True, nonnegative=True)
omega_y = sp.Symbol('omega_y', real=True, nonnegative=True)

# =============================================================================
# UTILITY AND OPTIMIZATION
# =============================================================================

U = sp.Symbol('U', real=True, positive=True)
lam = sp.Symbol('lambda', real=True, positive=True)  # Lagrange multiplier

# =============================================================================
# PARAMETERS
# =============================================================================

alpha = sp.Symbol('alpha', real=True, positive=True)
beta = sp.Symbol('beta', real=True, positive=True)
rho = sp.Symbol('rho', real=True)
gamma_x = sp.Symbol('gamma_x', real=True, nonnegative=True)
gamma_y = sp.Symbol('gamma_y', real=True, nonnegative=True)
