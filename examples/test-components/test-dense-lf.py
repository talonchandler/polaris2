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

xyzJ_list.data_xyz += np.array([0,0,0])
xyzJ_list.data_xyz *= 1.5

nvxulens = 3 # voxels per ulens side in object space
npxulens = 17 # pixels per ulens side in data space
nulens = 5 #13 # number of ulenses
z_slice = 45 

input_shape = (nvxulens*nulens,nvxulens*nulens,z_slice,6)
input_vox_dims = (17*6.5/60/nvxulens, 17*6.5/60/nvxulens, 0.2)
xyzJ = xyzJ_list.to_xyzJ(input_shape, input_vox_dims)
xyzJ.skip_n = 1
xyzJ.rad_scale = 1.5

# Testing
xyzJ.data[:,:,:,:] = 0
xyzJ.data[10,10,22,3] = 1
xyzJ.data[10,10,22,0] = 1

xyzJ.build_actors()

xyzJ.to_tiff('./guv.tiff')

# Grid peak
grid_peak = xyzJ.to_R3toR3_xyz()
grid_int = xyzJ.to_R3toR_xyz()
grid_peak.rad_scale = 1
grid_peak.build_actors()
grid_int.build_actors()

# Normal fourf
# d1 = det.FourF(npx=(101,101))
# im1 = d1.xyzJ_to_xy_det(grid_obj)
# im1.to_tiff('./4f.tiff')

# # LF detector
# d2 = det.FourFLF(irrad_title='Lightfield detector irradiance',
#                  npx=(npxulens*nulens, npxulens*nulens))
# d2.precompute_UvStzJ_to_UvSt_det(input_shape=(nulens,nvxulens,nulens,nvxulens,z_slice,6),
#                                  input_vox_dims=input_vox_dims)

# # Save
# import pickle
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(d2, handle)

# Load
with open('filename.pickle', 'rb') as handle:
    d2 = pickle.load(handle)

# Simulate
im2 = d2.xyzJ_to_xy_det(xyzJ)
im2.to_tiff('./out.tif')

# Reconstruct
xyzJ_recon = d2.pinv(im2, out_vox_dims=input_vox_dims)
xyzJ_recon.to_tiff('./recon.tif')

# Threshold
xyzJ_recon.threshold = 0.2*np.max(xyzJ_recon.data)

# Convert to other visuals
grid_peak_recon = xyzJ_recon.to_R3toR3_xyz()
grid_int_recon = xyzJ_recon.to_R3toR_xyz()

xyzJ_recon.build_actors()
grid_peak_recon.rad_scale = 3
grid_peak_recon.build_actors()
grid_int_recon.build_actors()

# Calculate im3
im3 = d2.xyzJ_to_xy_det(xyzJ_recon)
im3.to_tiff('./HH+Hf.tif')

# Titles
xyzJ.title = '$\mathbf{f}$'
grid_peak.title = 'Peaks$(\mathbf{f})$'
grid_int.title = 'MIP$(\mathbf{f})$'
im2.title = '$\mathcal{H}\mathbf{f}$'
xyzJ_recon.title = '$\mathcal{H^+H}\mathbf{f}$'
grid_peak_recon.title = 'Peaks$(\mathcal{H^+H}\mathbf{f})$'
grid_int_recon.title = 'MIP$(\mathcal{H^+H}\mathbf{f})$'
im3.title = '$\mathcal{HH^+H}\mathbf{f}$'

# Plot
obj_list = [xyzJ, grid_peak, grid_int, im2]
res_list = [xyzJ_recon, grid_peak_recon, grid_int_recon, im3]
utilmpl.plot([obj_list, res_list], './test-guv.png', ss=2)
