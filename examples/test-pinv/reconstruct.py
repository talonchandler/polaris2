import pickle
from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, phantoms, utilmpl
import logging
log = logging.getLogger('log')

# Set size of object and data
M = 60
wpx = 6.5
nvxulens = 3 # voxels per ulens side in object space
npxulens = 17 # pixels per ulens side in data space
nulens = 5 # number of ulenses
z_slice = 25

input_shape = np.array((nvxulens*nulens,nvxulens*nulens,z_slice,6))
input_vox_dims = np.array((npxulens*wpx/M/nvxulens, npxulens*wpx/M/nvxulens, 0.2))

# # Precompute detector (expensive)
# lfdet = det.FourFLF(npx=(npxulens*nulens, npxulens*nulens))
# lfdet.precompute_fwd(input_shape=(nulens,nvxulens,nulens,nvxulens,z_slice,6),
#                   input_vox_dims=input_vox_dims)
# lfdet.precompute_pinv()

# # Save detector
# with open('lfdet.pickle', 'wb') as handle:
#     pickle.dump(lfdet, handle)

# Load detector
with open('lfdet.pickle', 'rb') as handle:
    lfdet = pickle.load(handle)
import pdb; pdb.set_trace() 

# Main recon and plot loop
N = 30
for n in tqdm(range(N)):
    # f_list
    dir = phantoms.sphere_spiral(n/N)
    f_list = R3S2toR.xyzj_list([[0,0,1.5]], [dir], shape=input_shape[0:3]*input_vox_dims)
    f_list.rad_scale = 1.5
    f_list.build_actors()
    f_list.title = '$\mathbf{f}$'    

    # f
    f = f_list.to_xyzJ(xyzJ_shape=input_shape, vox_dims=input_vox_dims)

    # Peak(f)
    peakf = f.to_R3toR3_xyz()
    peakf.rad_scale = 1
    peakf.build_actors()
    peakf.title = 'Peaks$(\mathbf{f})$'

    # MIP(f)
    MIPf = f.to_R3toR_xyz()
    MIPf.build_actors()
    MIPf.title = 'MIP$(\mathbf{f})$'

    # g = Hf
    g = lfdet.fwd(f)
    g.title = '$\mathbf{g} = \mathcal{H}\mathbf{f}$'

    # f_hat = H^+Hf
    f_hat = lfdet.pinv(g, out_vox_dims=input_vox_dims)
    f_hat.threshold = 0.3*np.max(f_hat.data)
    f_hat.build_actors()
    f_hat.title = '$\hat{\mathbf{f}} = \mathcal{H^+H}\mathbf{f}$'

    # Peak(f_hat)
    peakf_hat = f_hat.to_R3toR3_xyz()
    peakf_hat.rad_scale = 1
    peakf_hat.build_actors()
    peakf_hat.title = 'Peaks$(\hat{\mathbf{f}})$'

    # MIP(f)
    MIPf_hat = f_hat.to_R3toR_xyz()
    MIPf_hat.build_actors()
    MIPf_hat.title = 'MIP$(\hat{\mathbf{f}})$'

    # g_hat = Hf_hat
    g_hat = lfdet.fwd(f_hat)
    g_hat.title = '$\hat{\mathbf{g}} = \mathcal{H}\hat{\mathbf{f}}$'
    
    obj_list = [f_list, peakf, MIPf, g]
    res_list = [f_hat, peakf_hat, MIPf_hat, g_hat]
    nstr = '{:03d}'.format(n)
    utilmpl.plot([obj_list, res_list], './out/'+nstr+'.png', ss=2)
