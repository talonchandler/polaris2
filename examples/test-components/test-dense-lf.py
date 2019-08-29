import pickle
from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

# Load guv
with open('../forward-defocus/guv-2um.pickle', 'rb') as handle:
    xyzJ_list = pickle.load(handle)

xyzJ_list.data_xyz += np.array([1,0,1])

nvxulens = 3 # voxels per ulens side in object space
npxulens = 17 # pixels per ulens side in data space
nulens = 7 # number of ulenses
z_slice = 31

input_shape = (nvxulens*nulens,nvxulens*nulens,z_slice,6)
input_vox_dims = (17*6.5/60/nvxulens, 17*6.5/60/nvxulens, 0.2)
xyzJ = xyzJ_list.to_xyzJ(input_shape, input_vox_dims)

xyzJ.build_actors()

xyzJ.to_tiff('./guv.tiff')

# Grid peak
grid_peak = xyzJ.to_R3toR3_xyz(skip_n=1)
grid_peak.rad_scale = 0.25
grid_peak.build_actors()

# Normal fourf
# d1 = det.FourF(npx=(101,101))
# im1 = d1.xyzJ_to_xy_det(grid_obj)
# im1.to_tiff('./4f.tiff')

# LF detector
d2 = det.FourFLF(irrad_title='Lightfield detector irradiance',
                 npx=(npxulens*nulens, npxulens*nulens))
d2.precompute_UvStzJ_to_UvSt_det(input_shape=(nulens,nvxulens,nulens,nvxulens,z_slice,6),
                                 input_vox_dims=input_vox_dims)

# Save
import pickle
with open('filename.pickle', 'wb') as handle:
    pickle.dump(d2, handle)

# Load
with open('filename.pickle', 'rb') as handle:
    d2 = pickle.load(handle)

# Simulate
im2 = d2.xyzJ_to_xy_det(xyzJ)
im2.data /= 100

obj_list = [xyzJ, grid_peak, im2]
utilmpl.plot([obj_list], './test-guv.png', ss=2)
