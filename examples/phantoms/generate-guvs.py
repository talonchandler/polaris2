import pickle
from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, R3toR, R3toR3, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

log.info('Generating phantom.')
# Oversampled object (create)
xyz_list, j_list = phantoms.guv(radius=2.0, ellip_ratio=0.2, M=2**13, dist_type='ellipsoid')
ss_obj = R3S2toR.xyzj_list(xyz_list, j_list, title='Dense object', rad_scale=0.125)

log.info('Converting phantom to grid.')
vox_dims = np.array([6.5/60,6.5/60,.2])
npx = np.array([45, 45, 25])
grid_obj = ss_obj.to_xyzJ(npx=npx, vox_dims=vox_dims, lmax=4)
grid_obj.to_tiff('./guv-2um.tiff')

grid_obj = R3S2toR.xyzJ(np.zeros((1,)), title='Dense object')
grid_obj.from_tiff('./guv-2um.tiff')

log.info('Calculating peaks')
peaks = grid_obj.to_R3toR3_xyz()

log.info('Calculating MIP')
mip = grid_obj.to_R3toR_xyz()

# Visualize
obj_list = [grid_obj, peaks, mip]
for obj in obj_list:
    obj.build_actors()

utilmpl.plot([obj_list], './visuals.png', ss=2)    

