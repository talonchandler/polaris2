from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilmpl, phantoms
import logging
log = logging.getLogger('log')

N = 2

log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):
    # xp, yp, zp = phantoms.sphere_spiral(i/(N-1))
    # print(xp,yp,zp)
    if i == 0:
        x, y, z = (0.5, 0, 0)
        xp, yp, zp = (1,0,1)
        
    else:
        x, y, z = (0, 0.5, 0)
        xp, yp, zp = (0,1,1)
        
    obj = R3S2toR.xyzj_list([[x,y,z]], [[xp,yp,zp]], shape=[10,10,4],
                            title='Single dipole radiator')

    obj.build_actors()

    d1 = det.FourFLF(ulens_aperture='square', irrad_title='Lightfield detector irradiance')
    im1 = d1.xyzj_single_to_xy_det(obj)
    im1.data /= 100

    d2 = det.FourF()
    im0 = d2.xyzj_single_to_xye_det(obj)
    im2 = d2.xyzj_single_to_xy_det(obj)
    im2.data /= 100

    istr = '{:03d}'.format(i)    
    im1.to_tiff('./out2/'+istr+'.tif')
    utilmpl.plot([[obj, im0, im2, im1]], './out/'+istr+'.png')
