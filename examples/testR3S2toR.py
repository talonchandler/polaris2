from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, R3toR, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

# Original test
# ellipsoid = np.zeros((100, 100, 40, 6))

# # Center
# ellipsoid[50,50,20,0] = 1
# ellipsoid[50,50,20,3] = 0.5

# # Increasing steps
# ellipsoid[0,10,20,0] = 1
# ellipsoid[20,10,20,0] = 2
# ellipsoid[40,10,20,0] = 3
# ellipsoid[60,10,20,0] = 4

# Nice plot
log.info('Generating phantom.')
vox_dims = np.array([.1,.1,.1])
npx = np.array([100, 100, 40])
xyz_list, j_list = phantoms.guv(radius=1.99, ellip_ratio=0.5)

obj = R3S2toR.xyzj_list(xyz_list, j_list,
                        shape=vox_dims*npx,
                        xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
                        title='Dense object',
                        skip_n=1, rad_scale=0.125)

# Oversampling
xyz_list, j_list = phantoms.guv(radius=1.99, ellip_ratio=0.5, M=2**10)
obj2 = R3S2toR.xyzj_list(xyz_list, j_list,
                        xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
                        title='Dense object',
                        skip_n=1, rad_scale=0.125)

log.info('Converting phantom to grid.')
obj3 = obj2.to_xyzJ(npx=npx, vox_dims=vox_dims, lmax=4)
obj3.rad_scale = 3
obj3.skip_n = 2

obj4 = R3toR.xyz(obj3.data[:,:,:,0],
                 vox_dims=vox_dims, 
                 xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
                 title='Maximum intensity projection')

obj.build_actors()
# obj3.build_actors()
obj4.build_actors()

# Flyaround
N = 80
log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):
    obj.increment_camera(360/N)
    obj3.increment_camera(360/N)
    obj4.increment_camera(360/N)
    istr = '{:03d}'.format(i)
    utilmpl.plot([[obj, obj4]], './test/'+istr+'.png', ss=1)
