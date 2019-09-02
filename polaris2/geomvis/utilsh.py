# Utilities for working with spherical harmponics
from scipy.special import sph_harm
from sympy import sqrt, I
from sympy.physics.wigner import gaunt, wigner_3j, clebsch_gordan
import numpy as np

# SciPy real spherical harmonics with identical interface to SymPy's Znm
# Useful for fast numerical evaluation of Znm
def spZnm(l, m, theta, phi):
    if m > 0:
        return np.sqrt(2)*((-1)**m)*np.real(sph_harm(m, l, phi, theta))
    elif m == 0:
        return np.real(sph_harm(m, l, phi, theta))
    elif m < 0:
        return np.sqrt(2)*((-1)**m)*np.imag(sph_harm(np.abs(m), l, phi, theta))

# Convert between spherical harmonic indices (l, m) and even order index (j)
def j2lm(j):
    if j < 0:
        return None
    l = 0
    while True:
        x = 0.5*l*(l+1)
        if abs(j - x) <= l:
            return l, int(j-x)
        else:
            l = l+2

def lm2j(l, m):
    if abs(m) > l or l%2 == 1:
        return None
    else:
        return int(0.5*l*(l+1) + m)

def maxl2maxj(l):
    return int(0.5*(l + 1)*(l + 2))

# Convert between Cartesian and spherical coordinates
def tp2xyz(tp):
    theta = tp[0]
    phi = tp[1]
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

def xyz2tp(x, y, z):
    arccos_arg = z/np.sqrt(x**2 + y**2 + z**2)
    if np.isclose(arccos_arg, 1.0): # Avoid arccos floating point issues
        arccos_arg = 1.0
    elif np.isclose(arccos_arg, -1.0):
        arccos_arg = -1.0
    return np.arccos(arccos_arg), np.arctan2(y, x)

# Returns "equally" spaced points on a unit sphere in spherical coordinates.
# http://stackoverflow.com/a/26127012/5854689
def fibonacci_sphere(n, xyz=False, pole=True):
    if pole:
        z = np.linspace(1, -1, num=n)
    else:
        z = np.linspace(1 - 1/n, -1 + 1/n, num=n) 
    theta = np.arccos(z)
    phi = np.mod((np.pi*(3.0 - np.sqrt(5.0)))*np.arange(n), 2*np.pi) - np.pi
    if xyz:
        return np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T
    else:
        return np.vstack((theta, phi)).T

# Calculate matrix that converts from N-point ODF to J-component SH coeffs
def calcB(N, J):
    tp = fibonacci_sphere(N)
    B = np.zeros((N, J))
    for (n, j), x in np.ndenumerate(B):
        l, m = j2lm(j)
        B[n, j] = spZnm(l, m, tp[n,0], tp[n,1])
    return B
    
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

# Calculate the Gaunt coefficient tensor for multiplying real even SH coeffs
def G_real_mult_tensor(Jout, Jin):
    G = np.zeros((Jout, Jin, Jin))
    for j in range(Jout):
        for jp in range(Jin):
            for jpp in range(Jin):
                l, m = j2lm(j)
                lp, mp = j2lm(jp)
                lpp, mpp = j2lm(jpp)
                G[j,jp,jpp] = gauntR(l,lp,lpp,m,mp,mpp)
    return G


# Compute Gaunt coefficient tensor for multiplying real SH coeffs in l=1 band
def gaunt_l1l1_tol0l2():
    # xyz2m = [-1,1,0]
    xyz2m = [1,-1,0]
    G = np.zeros((6, 3, 3))
    for i in range(6):
        for j in range(3):
            for k in range(3):
                l, m = j2lm(i)
                G[i,j,k] = gauntR(l,1,1,m,xyz2m[j],xyz2m[k])
    return G
