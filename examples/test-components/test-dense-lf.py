from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

# Load guv
# vox_dims = np.array([.1,.1,.1])
# npx = np.array([119, 119, 40])
grid_obj = R3S2toR.xyzJ(np.array([0]), title='Dense object')
grid_obj.from_tiff('../phantoms/guv-2um.tiff')
grid_obj.build_actors()

# npx = np.array([119, 119, 5])
# obj = np.zeros((npx[0], npx[1], npx[2], 6))
# obj[:,:,:,:6] = grid_obj.data[:,:,5:10,:6]

# grid_obj = R3S2toR.xyzJ(obj,
#                         vox_dims=vox_dims,
#                         title='Dense object',
#                         skip_n=2,
#                         rad_scale=5)
# grid_obj.build_actors()

grid_peak = grid_obj.to_R3toR3_xyz(skip_n=1)
grid_peak.rad_scale = 0.25
grid_peak.skip_n = 1
grid_peak.build_actors()

# Pad object
temp = np.zeros([101, 101, 25, 15])
temp[50-22:50+23,50-22:50+23,:,:] = grid_obj.data
grid_obj.data = temp

d1 = det.FourF(npx=(101,101))
im1 = d1.xyzJ_to_xy_det(grid_obj)
im1.to_tiff('./4f.tiff')

# grid_obj.data = grid_obj.data[0:51,0:51,0:10,:]
# d2 = det.FourFLF(irrad_title='Lightfield detector irradiance', npx=grid_obj.npx[0:2])
# d2.precompute_UvStzJ_to_UvSt_det(input_shape=(3,17,3,17,10,6),
#                                  input_vox_dims=(0.1,0.1,0.1))
# import pdb; pdb.set_trace()
# import pickle
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(d2, handle)

# with open('filename.pickle', 'rb') as handle:
#     d2 = pickle.load(handle)

# im2 = d2.xyzJ_to_xy_det(grid_obj)
# im2.data /= 100

obj_list = [grid_obj, grid_peak, im1]
utilmpl.plot([obj_list], './test-guv.png', ss=2)
