from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilmpl
import logging
log = logging.getLogger('log')

def sphere_spiral(t, a=3):
    tp = 2*t - 1
    c = np.sqrt(1 - tp**2)
    return c*np.cos(a*np.pi*tp), c*np.sin(a*np.pi*tp), tp

N = 80
log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):

    istr = '{:03d}'.format(i)
    xp, yp, zp = sphere_spiral(i/(N-1))
    obj = R3S2toR.xyzj_list([[0,0,1,xp,yp,zp]], shape=[10,10,4],
                            xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$', title='Single dipole radiator')
    obj.build_actors()

    d1 = det.FourFLF(ulens_aperture='square', irrad_title='Lightfield detector irradiance')
    im1 = d1.dip_to_det(obj)
    im1.data /= 100

    d2 = det.FourF()
    im2 = d2.dip_to_det(obj)
    im2.data /= 100

    # im1.save_tiff('./out2/'+istr+'.tif')
    utilmpl.plot([[obj, im2, im1]], './out/'+istr+'.png')
