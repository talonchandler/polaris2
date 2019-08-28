import pickle
from polaris2.micro.micro import det
import logging
log = logging.getLogger('log')
log.info('Generating microscope mapping matrix')

input_shape = (101,101,50,6)
input_vox_dims = (6.5/60, 6.5/60, 0.2)

det1 = det.FourF(npx=input_shape[0:2])
det1.precompute_XYzJ_to_XY_det(input_shape, input_vox_dims)

with open('det.pickle', 'wb') as handle:
    pickle.dump(det1, handle)
