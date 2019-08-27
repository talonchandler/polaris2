from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

# Load guv
vox_dims = np.array([.1,.1,.1])
npx = np.array([119, 119, 40])
grid_obj = R3S2toR.xyzJ(np.array([0]),
                        vox_dims=vox_dims,
                        xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
                        title='Dense object',
                        skip_n=2,
                        rad_scale=3)
grid_obj.from_tiff('./guv.tiff')
grid_obj.build_actors()

# npx = np.array([119, 119, 5])
# obj = np.zeros((npx[0], npx[1], npx[2], 6))
# obj[:,:,:,:6] = grid_obj.data[:,:,5:10,:6]

# grid_obj = R3S2toR.xyzJ(obj,
#                         vox_dims=vox_dims,
#                         xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
#                         title='Dense object',
#                         skip_n=2,
#                         rad_scale=5)
# grid_obj.build_actors()

grid_peak = grid_obj.to_R3toR3_xyz(skip_n=1)
grid_peak.rad_scale = 0.25
grid_peak.skip_n = 1
grid_peak.build_actors()

# d1 = det.FourF(npx=(119,119))
# im1 = d1.xyzJ_to_xy_det(grid_obj)
# im1.data /= 100

grid_obj.data = grid_obj.data
d2 = det.FourFLF(irrad_title='Lightfield detector irradiance', npx=npx[0:2])
im2 = d2.xyzJ_to_xy_det(grid_obj)
im2.data /= 100

obj_list = [grid_obj, grid_peak, im2]
utilmpl.plot([obj_list], './test.png', ss=2)

# N = 60
# log.info('Making '+str(N)+' frames')
# for i in tqdm(range(N)):
#     for obj in [grid_obj, grid_peak]:
#         obj.increment_camera(360/N)
#     istr = '{:03d}'.format(i)
#     utilmpl.plot([obj_list], './test/'+istr+'.png', ss=2)
