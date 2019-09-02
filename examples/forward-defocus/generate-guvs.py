import pickle
from tqdm import tqdm
import numpy as np
from polaris2.micro.micro import det
from polaris2.geomvis import R3S2toR, R3toR, R3toR3, utilsh, utilmpl, phantoms
import logging
log = logging.getLogger('log')

log.info('Generating phantom.')
# Oversampled object (create)
xyz_list, j_list = phantoms.guv(radius=2.0, ellip_ratio=0.2, M=2**17, dist_type='ellipsoid')
guv_xyzj_list = R3S2toR.xyzj_list(xyz_list, j_list, title='Dense object', rad_scale=0.125)
guv_xyzJ_list = guv_xyzj_list.to_xyzJ_list()

with open('guv-2um.pickle', 'wb') as handle:
    pickle.dump(guv_xyzJ_list, handle)

