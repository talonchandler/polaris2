from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilmpl, phantoms
import logging
log = logging.getLogger('log')

N = 10 

log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):
    x, y, z = 0, 0, 0 # dipole position wrt in-focus origin in um (change this to see defocused or between-microlens PSFs)
    xp, yp, zp = phantoms.sphere_spiral(i/(N-1)) # dipole orientation (no need to normalize)
        
    obj = R3S2toR.xyzj_list([[x,y,z]], [[xp,yp,zp]], shape=[10,10,4],
                            title='Single dipole radiator')

    obj.build_actors()

    # Generate and simulate default lightfield setup (see polaris2/micro/det.py)
    LFdet = det.FourFLF(NA=1.2,
                        M=60, 
                        n0=1.3, # index of refraction
                        lamb=0.546, # wavelength (um)
                        npx=(7*17, 7*17), # number of pixels on detector
                        ss=2, # super-sample factor for each pixel (higher is more accurate)
                        wpx_real=6.5, # pixel width in object space (um)
                        plotfov=10, # detector region plotted
                        fulen=2500, # ulens focal length
                        ulenpx=17, # pixels behind each ulens
                        ulens_aperture='square', # also accepts 'circle'
                        irrad_title='Lightfield detector irradiance')
    
    LFdet_irr = LFdet.xyzj_single_to_xy_det(obj)
    LFdet_irr.data /= 100

    # Generate and simulate default 4F setup (see polaris2/micro/det.py class FourF for default params)
    det4f = det.FourF()
    det_e = det4f.xyzj_single_to_xye_det(obj)
    det_irr = det4f.xyzj_single_to_xy_det(obj)
    det_irr.data /= 100

    # Generate visualizations
    istr = '{:03d}'.format(i)    
    LFdet_irr.to_tiff('./lf-det-tifs/'+istr+'.tif')
    utilmpl.plot([[obj, det_e, det_irr, LFdet_irr]], './results/'+istr+'.png')
