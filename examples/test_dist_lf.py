from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

# N = 40
# for n in tqdm(range(14, N)):
#     # Plot basis functions
#     objs = []
#     ims = []
#     ims2 = []
#     x, y, z = phantoms.defocus_path(n/N)
#     for i in range(6):
#         J = np.zeros(6)
#         J[i] = 1
#         l, m = utilsh.j2lm(i)
#         obj = R3S2toR.xyzJ_single([x,y,z,J], shape=[10,10,4],
#                                   xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
#                                   title='$\ell='+str(l)+', m='+str(m)+'$')
#         obj.build_actors()
#         objs.append(obj)

#         d = det.FourF()
#         im = d.xyzJ_single_to_xy_det(obj)
#         im.cmap = 'bwr'
#         ims.append(im)

#         d2 = det.FourFLF()
#         im2 = d2.xyzJ_single_to_xy_det(obj)
#         im2.cmap = 'bwr'
#         ims2.append(im2)
        
#     istr = '{:03d}'.format(n)
#     utilmpl.plot([objs, ims, ims2], './basis_functions/'+istr+'.png')

# Ellipsoid spiral
N = 80
log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):

    pos = phantoms.defocus_path(i/N)
    pos2 = pos + np.array([2.5,0,0])
    ss = phantoms.sphere_spiral(i/(N-1))

    jj = phantoms.uniaxial_ellipsoid(1, 0.1, ss)

    obj = R3S2toR.xyzj_list([pos, pos2], [jj, jj], shape=[10,10,4],
                            xlabel='10$\\times$10$\\times$4 $\mu$m${}^3$',
                            title='Uniaxial distribution $a/b = 0.1$')

    obj2 = obj.to_xyzJ_list(lmax=2)
    obj2.title = '$\ell = 2$ projection'

    obj.build_actors()
    obj2.build_actors()    

    d1 = det.FourF()
    im1 = d1.xyzJ_list_to_xy_det(obj2)
    im1.data /= 100

    d2 = det.FourFLF(irrad_title='Lightfield detector irradiance')
    im2 = d2.xyzJ_list_to_xy_det(obj2)
    im2.data /= 100

    istr = '{:03d}'.format(i)
    utilmpl.plot([[obj, im1, im2]], './test/'+istr+'.png')
