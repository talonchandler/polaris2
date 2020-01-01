import pickle
from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, R3toR, R3toR3, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

log.info('Generating phantom.')
vox_dims = np.array([.1,.1,.1])

data = np.zeros((3,3,3,6))
data[0,0,0,0] = 1
data[1,1,1,0] = 1
data[2,2,2,0] = 1
data[2,0,0,0] = 1

grid_obj = R3S2toR.xyzJ(data, vox_dims=vox_dims)
grid_obj.to_tiff('./guv.tiff')
grid_obj.rad_scale = 0.5

log.info('Calculating MIP')
mip = grid_obj.to_R3toR_xyz()


obj_list = [grid_obj, mip]
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
