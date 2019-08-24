import numpy as np
from polaris2.geomvis import utilsh

# Returns x, y, z coordinates of a path that:
# 1) Starts at the origin
# 2) Moves along +z by d_max
# 3) Moves along (x, y) by d_max
# 4) Moves along -z by 2*d_max
# 5) Moves along (-x, -y) by d_max
# 6) Moves along +z by d_max back to the origins
def defocus_path(t, d_max=2):
    if t < 1/6:
        return 0, 0, 6*t*d_max
    elif t < 2/6:
        return 6*(t-1/6)*d_max, 6*(t-1/6)*d_max, d_max
    elif t < 4/6:
        return d_max, d_max, d_max-6*(t-2/6)*d_max
    elif t < 5/6:
        return d_max-6*(t-4/6)*d_max, d_max-6*(t-4/6)*d_max, -d_max
    elif t < 1:
        return 0, 0, -d_max + 6*(t-5/6)*d_max
    else:
        return 0, 0, 0
    
# Returns x, y, z coordinates of a spiral on the unit sphere 
# Sweep through spiral with t in [0, 1]
# rev is the number of azimuthal revolutions
def sphere_spiral(t, rev=3):
    tp = 2*t - 1
    c = np.sqrt(1 - tp**2)
    return c*np.cos(rev*np.pi*tp), c*np.sin(rev*np.pi*tp), tp

# Return point-wise representation of an ellipsoid with inputs:
# principal radii [r1, r2, r3], and 
# principal axes [[x1, y1, z1], [x2, y2, z3], [x3, y3, z3]].
# principal axes should be orthonormal
def ellipsoid(pradii, paxes, N=2**12):
    xyz = utilsh.fibonacci_sphere(N, xyz=True)
    xyz_rot = np.einsum('ij,kj->ik', xyz, paxes) # Rotate
    return 1/np.linalg.norm(xyz_rot/np.array(pradii), ord=2, axis=-1)

# Return point-wise representation of a uniaxial ellipsoid with inputs:
# uniaxial radius r1
# perpedicular radii r2
# uniaxial direction v1 = [x1, y1, z1]
def uniaxial_ellipsoid(r1, r2, v1, N=2**12):
    v1 = np.array(v1)
    v2 = np.random.randn(3)
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return ellipsoid([r1, r2, r2], np.array([v1, v2, v3]), N=N)

# Returns an xyzj_list array representing a guv with
# npx = [x, y, z] voxels with dimensions vox_dims = [xd, yd, zd] with radius
# centered at center = [xc, yc, zc] up to 
#
# The routine "voxelizes" the sphere by integrating over M points on the sphere.
def guv(center=[0,0,0], radius=0.5, N=2**10, M=2**10, ellip_ratio=0.5,
        dist_type='ellipsoid'):
    xyz_guv = utilsh.fibonacci_sphere(M, xyz=True) # Points on guv
    xyz_list = (xyz_guv*radius + center) # Scaled and center
    j_list = np.zeros((M, N))
    for m in range(M):
        if dist_type == 'ellipsoid':
            j_list[m,:] = uniaxial_ellipsoid(1, ellip_ratio, xyz_guv[m], N=N)
        elif dist_type == 'uniform':
            j_list[m,:] = np.ones((N,))
    return xyz_list, j_list

