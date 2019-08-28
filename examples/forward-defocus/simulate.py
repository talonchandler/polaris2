import pickle
from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, R3toR, R3toR3, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

# Load object
with open('guv-2um.pickle', 'rb') as handle:
    xyzJ_list = pickle.load(handle)

# Load microscope
with open('det.pickle', 'rb') as handle:
    det1 = pickle.load(handle)

xyzJ_list.data_J[:,1:] = 0 # Test

xyzJ_list.data_xyz += np.array([0,0,-2.5])
N = 50
for i in range(N):
    # Shift up
    xyzJ_list.data_xyz += np.array([0,0,0.1])

    # Put object on a grid
    input_shape = (101,101,50,6)
    input_vox_dims = (6.5/60, 6.5/60, 0.2)
    xyzJ = xyzJ_list.to_xyzJ(input_shape, input_vox_dims)

    # Simulate microscope
    xy = det1.xyzJ_to_xy_det(xyzJ)

    # Plot
    xyzJ.skip_n = 2
    xyzJ.rad_scale = 1.5
    xyzJ.build_actors()
    obj_list = [xyzJ, xy]
    utilmpl.plot([obj_list], './out/'+str(i)+'.png', ss=2)
