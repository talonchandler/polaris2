import pickle
from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, R3toR, R3toR3, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

log.info('Generating phantom.')
vox_dims = np.array([.1,.1,.2])
npx = np.array([41, 41, 21])

# Object for visualization
xyz_list, j_list = phantoms.guv(radius=2, ellip_ratio=0.3, M=2**10)
viz_obj = R3S2toR.xyzj_list(xyz_list, j_list,
                            shape=vox_dims*npx,
                            title='Dense object',
                            rad_scale=0.125)
peaks = viz_obj.to_R3toR3_xyz(shape=vox_dims*npx)

# Oversampled object (create)
xyz_list, j_list = phantoms.guv(radius=1.9, ellip_ratio=0.3, M=2**14, dist_type='ellipsoid')
ss_obj = R3S2toR.xyzj_list(xyz_list, j_list,
                           title='Dense object',
                           rad_scale=0.125)
log.info('Converting phantom to grid.')
grid_obj = ss_obj.to_xyzJ(xyzJ_shape=[41,41,21,6], vox_dims=vox_dims)
grid_obj.to_tiff('./guv.tiff')


# Load guv
grid_obj = R3S2toR.xyzJ(np.array([0]),
                        vox_dims=vox_dims,
                        title='Dense object',
                        skip_n=2,
                        rad_scale=2.0)
grid_obj.from_tiff('./guv.tiff')

log.info('Calculating MIP')
mip = grid_obj.to_R3toR_xyz()

obj_list = [viz_obj, peaks, mip, grid_obj]
for obj in obj_list:
    obj.build_actors()

# Flyaround
N = 80
log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):
    for obj in obj_list:
        obj.increment_camera(360/N)
    istr = '{:03d}'.format(i)
    utilmpl.plot([obj_list], './test/'+istr+'.png', ss=2)
