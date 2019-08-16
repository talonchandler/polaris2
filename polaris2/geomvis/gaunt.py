# Tools for multiplying real spherical harmonics

# Loosely based on:
# Herbert H.H. Homeier, E. Otto Steinborn,
# Some properties of the coupling coefficients of real spherical harmonics
# and their relation to Gaunt coefficients,
# Journal of Molecular Structure
# Volume 368, 1996, Pages 31-37, ISSN 0166-1280,
# https://doi.org/10.1016/S0166-1280(96)90531-X.

# Note that Homeier and Steinborn use the following definition of the real
# spherical harmonics:
#
#           / \sqrt(2) Re(Y_l^m(s)) for m > 0
# Y_lm(s) = | Y_l^0(s) for m = 0
#           \ \sqrt(2) Im(Y_l^m(s)) for m < 0,
#
# while here we use the following definition (from Wikipedia, Jarosz). 
#
#           / \sqrt(2) (-1)^m Re(Y_l^m(s)) for m > 0
# Y_lm(s) = | Y_l^0(s) for m = 0
#           \ \sqrt(2) (-1)^m Im(Y_l^m(s)) for m < 0.
#
# Note that other definitions exist (sympy's Znm has its own definition).
# USER BEWARE!

from sympy import sqrt, I
from sympy.physics.wigner import gaunt, wigner_3j, clebsch_gordan

# Unitary matrix that transforms complex sh to real sh
# Y_lm(s) = \sum_mp U_{l,m,mp} Y_l^mp(s)
def U(l, m, mp):
    if m > abs(l) or mp > abs(l): # outside matrix
        return 0
    elif abs(m) == abs(mp): # on the diagonals
        if m == 0 and mp == 0: # in the center
            return 1
        # four diagonals
        elif m > 0 and mp > 0:
            return (-1)**m/sqrt(2)
        elif m > 0 and mp < 0:
            return 1/sqrt(2)
        elif m < 0 and mp > 0:
            return -(-1)**m*I/sqrt(2)
        elif m < 0 and mp < 0:
            return I/sqrt(2)
    else:
        return 0

# Real gaunt coefficients
# See Eqs. 26. 
# This sum could be truncated using selection rules, but this is fast enough.
def gauntR(l1, l2, l3, m1, m2, m3, evaluate=True):
    result = 0
    for m1p in range(-l1, l1+1):
        U1 = U(l1, m1, m1p)
        for m2p in range(-l2, l2+1):
            U2 = U(l2, m2, m2p)
            for m3p in range(-l3, l3+1):
                U3 = U(l3, m3, m3p)
                result += U1*U2*U3*gaunt(l1, l2, l3, m1p, m2p, m3p)
    if evaluate:
        return float(result)
    else:
        return result
