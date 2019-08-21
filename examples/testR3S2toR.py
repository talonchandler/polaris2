from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')


# ellipsoid = phantoms.guv(radius=0.5)

# Original test
ellipsoid = np.zeros((100, 100, 40, 6))

# Center
ellipsoid[50,50,20,0] = 1
ellipsoid[50,50,20,3] = 0.5

# Increasing steps
ellipsoid[0,10,20,0] = 1
ellipsoid[20,10,20,0] = 2
ellipsoid[40,10,20,0] = 3
ellipsoid[60,10,20,0] = 4

obj = R3S2toR.xyzJ(ellipsoid, vox_dims=[.1,.1,.1],
                   xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
                   title='Dense object',
                   max_in_rad=2, max_out_rad=0.5)
obj.build_actors()

# Flyaround
N = 80
log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):
    obj.increment_camera(360/N)
    istr = '{:03d}'.format(i)
    utilmpl.plot([[obj]], './test/'+istr+'.png')
