from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

# N = 40
# for n in tqdm(range(N)):
#     # Plot basis functions
#     objs = []
#     ims = []
#     ims2 = []
#     x, y, z = phantoms.defocus_path(n/N)
#     istr = '{:03d}'.format(n)    
#     for i in range(6):
#         J = np.zeros(6)
#         J[i] = 1
#         l, m = utilsh.j2lm(i)

#         if n == 0:
#             obj = R3S2toR.xyzJ_list([[0.5,0,0]], [J], shape=[10,10,4],
#                                       title='$\ell='+str(l)+', m='+str(m)+'$')
#         if n == 1:
#             obj = R3S2toR.xyzJ_list([[0,0.5,0]], [J], shape=[10,10,4],
#                                       title='$\ell='+str(l)+', m='+str(m)+'$')
        
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
#         im2.to_tiff('./basis_out/'+istr+'-'+str(i)+'.tif')

#     utilmpl.plot([objs, ims, ims2], './basis_functions/'+istr+'.png')
    

# Ellipsoid spiral
N = 40
log.info('Making '+str(N)+' frames')
for i in tqdm(range(N)):

    pos = np.array([0,0,0])#phantoms.defocus_path(i/N)
    ss = phantoms.sphere_spiral(i/(N-1))

    jj = phantoms.uniaxial_ellipsoid(1,0.0001, ss, N=2**14)

    obj = R3S2toR.xyzj_list([pos], [jj], shape=[10,10,4],
                            title='Uniaxial distribution $a/b = 0.1$')

    obj2 = obj.to_xyzJ_list(Jmax=6)
    obj2.title = '$\ell = 2$ projection'
    print(obj2.data_J)

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
    # im1.to_tiff('./test1/'+istr+'.tif')
    # im2.to_tiff('./test2/'+istr+'.tif')
